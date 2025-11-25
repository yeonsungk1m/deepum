import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import torch

############################
# New helper functions
############################

def load_ca_coordinates(pdb_path):
    """
    Parse a PDB file and return C-alpha coordinates as an (N, 3) array.

    We assume that all models (native and decoys) share the same residue
    ordering so we don't try to do any complex sequence alignment here.
    """
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            # Standard PDB ATOM record, C-alpha atom
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords.append((x, y, z))
    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return np.asarray(coords, dtype=np.float32)


def compute_lddt_ca(native_ca, model_ca, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    Very small, self-contained implementation of lDDT-Ca.

    native_ca, model_ca : (N,3) arrays of CA coordinates for native and model.
    cutoff : inclusion radius in Angstroms
    thresholds : tuple of distance-difference thresholds (Å)

    Returns
    -------
    per_res_lddt : (N,) array with lDDT per residue in [0,1].
                   Residues with no neighbors inside cutoff get np.nan.
    """
    if native_ca.shape != model_ca.shape:
        raise ValueError(f"Native/model CA shape mismatch: {native_ca.shape} vs {model_ca.shape}")

    N = native_ca.shape[0]

    # Pairwise CA-CA distances
    diff_native = native_ca[:, None, :] - native_ca[None, :, :]
    dist_native = np.linalg.norm(diff_native, axis=-1)

    diff_model = model_ca[:, None, :] - model_ca[None, :, :]
    dist_model = np.linalg.norm(diff_model, axis=-1)

    # We ignore self distances (i==j) and only keep neighbors within cutoff
    contact_mask = (dist_native > 0.0) & (dist_native <= cutoff)

    # Absolute distance difference between model and native
    diff = np.abs(dist_model - dist_native)  # (N,N)

    thr = np.asarray(thresholds, dtype=np.float32).reshape(len(thresholds), 1, 1)
    # good[t, i, j] = True if |d_model - d_native| < thresholds[t] AND within cutoff
    good = (diff[None, :, :] < thr) & contact_mask[None, :, :]

    # For each residue i, count contacts over j within cutoff
    contact_counts = contact_mask.sum(axis=1).astype(np.float32)  # (N,)

    # Sum over thresholds and neighbors j
    good_counts = good.sum(axis=(0, 2)).astype(np.float32)  # (N,)

    # Avoid division by zero; residues with no contacts will be set to NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        per_res = good_counts / (len(thresholds) * contact_counts)
    per_res[contact_counts == 0] = np.nan

    return per_res  # (N,)


def pearsonr_np(x, y):
    """Pure-numpy Pearson correlation (returns float or np.nan)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        return np.nan
    xm = x.mean()
    ym = y.mean()
    x_centered = x - xm
    y_centered = y - ym
    num = np.sum(x_centered * y_centered)
    den = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if den == 0.0:
        return np.nan
    return float(num / den)


def _rankdata_dense(a):
    """
    Dense ranking: smallest value -> rank 0, next -> rank 1, ...
    Ties get the same rank. This is sufficient for Spearman here.
    """
    a = np.asarray(a)
    if a.size == 0:
        return a.astype(float)
    _, inv = np.unique(a, return_inverse=True)
    return inv.astype(np.float64)


def spearmanr_np(x, y):
    """Spearman rank correlation via dense ranks + Pearson."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size or x.size == 0:
        return np.nan
    rx = _rankdata_dense(x)
    ry = _rankdata_dense(y)
    return pearsonr_np(rx, ry)


def _rankdata_average(a):
    """
    Rank data with average ranks for ties, 1-based ranks.
    Used for AUC via the Mann–Whitney formulation.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        # entries order[i:j] share the same value
        # positions (1-based) are i+1, ..., j
        rank_val = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank_val
        i = j
    return ranks


def binary_auc(y_true, y_score):
    """
    ROC AUC from binary labels and scores, implemented with pure numpy.

    Parameters
    ----------
    y_true : array-like of shape (N,), elements in {0,1}
    y_score : array-like of shape (N,), predicted scores (higher=better)

    Returns
    -------
    auc : float in [0,1] or np.nan if undefined
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_true.size != y_score.size:
        return np.nan

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = _rankdata_average(y_score)  # 1-based average ranks for ties
    rank_sum_pos = ranks[pos_mask].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def evaluate_against_native(input_dir, pred_global, pred_local, auc_threshold=0.6, verbose=False):
    """
    Compute evaluation metrics given predicted lDDT (global + per-residue)
    and a native structure named 'native.pdb' in input_dir.

    Parameters
    ----------
    input_dir : str
        Folder containing native.pdb and decoy PDBs.
    pred_global : dict
        {sample_name(str) -> predicted global lDDT (float)}.
        sample_name does NOT include '.pdb'.
    pred_local : dict
        {sample_name(str) -> np.ndarray of shape (L,) with predicted per-residue lDDT}.
    auc_threshold : float
        Threshold on **true** lDDT to define "good" vs "bad" for AUC.
    """
    native_path = join(input_dir, "native.pdb")
    if not isfile(native_path):
        print("[eval] native.pdb not found in input directory; skip evaluation.", file=sys.stderr)
        return

    try:
        native_ca = load_ca_coordinates(native_path)
    except Exception as e:
        print(f"[eval] Failed to load native CA from {native_path}: {e}", file=sys.stderr)
        return

    global_true = []
    global_pred = []
    local_true_list = []
    local_pred_list = []

    # treat everything except 'native' as decoy
    for sample_name, g_pred in pred_global.items():
        if sample_name.lower() == "native":
            continue

        decoy_pdb = join(input_dir, sample_name + ".pdb")
        if not isfile(decoy_pdb):
            if verbose:
                print(f"[eval] Decoy PDB not found, skip: {decoy_pdb}", file=sys.stderr)
            continue

        if sample_name not in pred_local:
            if verbose:
                print(f"[eval] No local prediction for sample {sample_name}, skip.", file=sys.stderr)
            continue

        try:
            model_ca = load_ca_coordinates(decoy_pdb)
        except Exception as e:
            if verbose:
                print(f"[eval] Failed to load CA for {decoy_pdb}: {e}", file=sys.stderr)
            continue

        if model_ca.shape != native_ca.shape:
            if verbose:
                print(f"[eval] Length mismatch (native {native_ca.shape[0]}, model {model_ca.shape[0]}), skip {sample_name}", file=sys.stderr)
            continue

        # ground-truth per-residue lDDT (lDDT-Ca approximation)
        true_lddt = compute_lddt_ca(native_ca, model_ca)  # (L,)
        pred_lddt = np.asarray(pred_local[sample_name], dtype=np.float32).reshape(-1)

        L = min(true_lddt.size, pred_lddt.size)
        true_lddt = true_lddt[:L]
        pred_lddt = pred_lddt[:L]

        # drop residues where GT is NaN (no contacts)
        mask = ~np.isnan(true_lddt)
        if not np.any(mask):
            continue

        true_lddt = true_lddt[mask]
        pred_lddt = pred_lddt[mask]

        global_true.append(float(true_lddt.mean()))
        global_pred.append(float(g_pred))

        local_true_list.append(true_lddt)
        local_pred_list.append(pred_lddt)

    if len(global_true) == 0:
        print("[eval] No valid decoys after filtering; cannot compute metrics.", file=sys.stderr)
        return

    global_true = np.concatenate([np.asarray(global_true, dtype=np.float32)])
    global_pred = np.concatenate([np.asarray(global_pred, dtype=np.float32)])
    local_true = np.concatenate(local_true_list).astype(np.float32)
    local_pred = np.concatenate(local_pred_list).astype(np.float32)

    # Global metrics
    g_pear = pearsonr_np(global_true, global_pred)
    g_spear = spearmanr_np(global_true, global_pred)
    g_mae = float(np.mean(np.abs(global_pred - global_true)))
    g_labels = (global_true >= auc_threshold).astype(np.int32)
    g_auc = binary_auc(g_labels, global_pred)

    # Local metrics (per-residue, pooled over all residues)
    l_pear = pearsonr_np(local_true, local_pred)
    l_spear = spearmanr_np(local_true, local_pred)
    l_mae = float(np.mean(np.abs(local_pred - local_true)))
    l_labels = (local_true >= auc_threshold).astype(np.int32)
    l_auc = binary_auc(l_labels, local_pred)

    print("==== DeepUMQA evaluation against native.pdb ====")
    print(f"#decoys used: {len(global_true)}")
    print(f"#residues (pooled): {local_true.size}")
    print("")
    print("Global QA metrics (per-model):")
    print(f"  Pearson r      : {g_pear:.4f}")
    print(f"  Spearman rho   : {g_spear:.4f}")
    print(f"  MAE (|pred-gt|): {g_mae:.4f}")
    print(f"  AUC (thr={auc_threshold:.2f}): {g_auc:.4f}")
    print("")
    print("Local QA metrics (per-residue, pooled):")
    print(f"  Pearson r      : {l_pear:.4f}")
    print(f"  Spearman rho   : {l_spear:.4f}")
    print(f"  MAE (|pred-gt|): {l_mae:.4f}")
    print(f"  AUC (thr={auc_threshold:.2f}): {l_auc:.4f}")


############################
# Original main script (+ eval option)
############################

def main():

    parser = argparse.ArgumentParser(description="predictor network error")
    parser.add_argument("input",
                        action="store",
                        help="path to input ")
    
    parser.add_argument("output",
                        action="store", nargs=argparse.REMAINDER,
                        help="path to output")
    
    parser.add_argument("--pdb",
                        "-pdb",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb ")
    
    parser.add_argument("--csv",
                        "-csv",
                        action="store_true",
                        default=False,
                        help="Writing results to a csv file ")

    parser.add_argument("--per_res_only",
                        "-pr",
                        action="store_true",
                        default=False,
                        help="Store per-residue accuracy only")
    
    parser.add_argument("--leaveTempFile",
                        "-lt",
                        action="store_true",
                        default=False,
                        help="Leaving temporary files")
    
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="Specifying # of cpus to use for featurization")
    
    parser.add_argument("--featurize",
                        "-f",
                        action="store_true",
                        default=False,
                        help="Running only the featurization part")
    
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="Reprocessing all feature files")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag ")
    
    
    parser.add_argument("--ensemble",
                        "-e", 
                        action="store_true",
                        default=False,
                        help="Running with ensembling of 4 models. ")

    # NEW: evaluation switch + AUC threshold
    parser.add_argument("--eval",
                        action="store_true",
                        default=False,
                        help="Compute evaluation metrics vs native.pdb in input folder")
    parser.add_argument("--auc_threshold",
                        type=float,
                        default=0.6,
                        help="Ground-truth lDDT threshold used to define 'good' vs 'bad' for AUC (default: 0.6)")
    
    args = parser.parse_args()

    csvfilename = "result.csv"

    if len(args.output)>1:
        print(f"Only one output folder can be specified, but got {args.output}", file=sys.stderr)
        return -1
    
    if len(args.output)==0:
        args.output = ""
    else:
        args.output = args.output[0]

    if args.input.endswith('.pdb'):
        args.pdb = True
    
    if args.output.endswith(".csv"):
        args.csv = True
        
    if not args.pdb:
        if not isdir(args.input):
            print("Input folder does not exist.", file=sys.stderr)
            return -1

        if args.output == "":
            args.output = args.input
        else:
            if not args.csv and not isdir(args.output):
                if args.verbose: print("Creating output folder:", args.output)
                os.mkdir(args.output)
            
            # if csv, do it in place.
            elif args.csv:
                csvfilename = args.output
                args.output = args.input
          
    else:
        if not isfile(args.input):
            print("Input file does not exist.", file=sys.stderr)
            return -1

        if args.output == "":
            args.output = os.path.splitext(args.input)[0]+".npz"

        if not(".pdb" in args.input and ".npz" in args.output):
            print("Input needs to be in .pdb format, and output needs to be in .npz format.", file=sys.stderr)
            return -1
        
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "model/")
    
    modelpath = join(base, "DeepUMQA")


    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import deepUMQA as umqa
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
        

    if not args.pdb:
        samples = [i[:-4] for i in os.listdir(args.input) if isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!="."]
        ignored = [i[:-4] for i in os.listdir(args.input) if not(isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
        if args.verbose: 
            print("# samples:", len(samples))
            if len(ignored) > 0:
                print("# files ignored:", len(ignored))


        inputs = [join(args.input, s)+".pdb" for s in samples]
        tmpoutputs = [join(args.output, s)+".features.npz" for s in samples]
        
        if not args.reprocess:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are already processed.")
        else:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are re-processed.")

        if num_process == 1:
            for a in arguments:
                umqa.process(a)
        else:
            pool = multiprocessing.Pool(num_process)
            out = pool.map(umqa.process, arguments)
            

        if args.featurize:
            return 0
        
        if args.verbose: print("using", modelpath)

        
        samples = [s for s in samples if isfile(join(args.output, s+".features.npz"))]

        if args.ensemble:
            modelnames = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl"]
        else:
            modelnames = ["best.pkl"]
        
        result = {}
        # for evaluation: we keep local predictions per-sample, per-model
        eval_local_raw = {}  # sample -> list of np.ndarray (one per model)

        for modelname in modelnames:
            if args.eval and args.ensemble:
                if args.verbose:
                    print(f"[eval] Running evaluation with ensemble member {modelname}")

            model = umqa.myDeepUMQA(twobody_size = 33)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(join(modelpath, modelname), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            for s in samples:
                #try:
                    with torch.no_grad():
                        if args.verbose: print("Predicting for", s) 
                        filename = join(args.output, s+".features.npz")
                        bertname = ""
                        (idx, val), (f1d, bert), f2d, dmy = umqa.getData(filename, bertpath = bertname)
                        f1d = torch.Tensor(f1d).to(device)
                        f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
                        idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                        val = torch.Tensor(val).to(device)

                        deviation, mask, lddt, dmy = model(idx, val, f1d, f2d)

                        lddt_np = lddt.cpu().detach().numpy().astype(np.float32).reshape(-1)
                        g_score = float(lddt_np.mean())

                        t = result.get(s, [])
                        t.append(g_score)
                        result[s] = t

                        if args.eval:
                            # collect per-residue predictions for later averaging
                            lst = eval_local_raw.get(s, [])
                            lst.append(lddt_np)
                            eval_local_raw[s] = lst

                        if not args.csv:
                            if args.ensemble:
                                if args.per_res_only:
                                    np.savez_compressed(join(args.output, s+"_"+modelname[:-4]+".npz"),
                                                        lddt = lddt_np.astype(np.float16))
                                else:
                                    np.savez_compressed(join(args.output, s+"_"+modelname[:-4]+".npz"),
                                                        lddt = lddt_np.astype(np.float16),
                                                        deviation = deviation.cpu().detach().numpy().astype(np.float16),
                                                        mask = mask.cpu().detach().numpy().astype(np.float16))
                            else:
                                if args.per_res_only:
                                    np.savez_compressed(join(args.output, s+".npz"),
                                                        lddt = lddt_np.astype(np.float16))
                                else:
                                    np.savez_compressed(join(args.output, s+".npz"),
                                                        lddt = lddt_np.astype(np.float16),
                                                        deviation = deviation.cpu().detach().numpy().astype(np.float16),
                                                        mask = mask.cpu().detach().numpy().astype(np.float16))
                #except:
                    #print("Failed to predict for", join(args.output, s+"_"+modelname[:-4]+".npz"))
        
        # Aggregate predictions across ensemble members
        pred_global = {}
        for s in samples:
            if s in result:
                pred_global[s] = float(np.mean(result[s]))

        pred_local = {}
        if args.eval:
            for s, arrs in eval_local_raw.items():
                if len(arrs) == 0:
                    continue
                # align lengths in case they differ slightly
                min_len = min(a.shape[0] for a in arrs)
                arrs_trim = [a[:min_len] for a in arrs]
                pred_local[s] = np.mean(np.stack(arrs_trim, axis=0), axis=0).astype(np.float32)

            # Actually compute metrics vs native.pdb
            evaluate_against_native(args.input, pred_global, pred_local,
                                    auc_threshold=args.auc_threshold,
                                    verbose=args.verbose)
        
        if not args.csv:            
            if args.ensemble:
                umqa.merge(samples, args.output, verbose=args.verbose)
            
            if not args.leaveTempFile:
                umqa.clean(samples,
                          args.output,
                          verbose=args.verbose,
                          ensemble=args.ensemble)
        else:
            # Take average of outputs
            csvfile = open(csvfilename, "w")
            csvfile.write("sample\tcb-lddt\n")
            for s in samples:
                if s in pred_global:
                    line = "%s\t%.4f\n"%(s, pred_global[s])
                    csvfile.write(line)
            csvfile.close()
            
    # Processing for single sample
    else:
        infilepath = args.input
        outfilepath = args.output
        infolder = "/".join(infilepath.split("/")[:-1])
        insamplename = infilepath.split("/")[-1][:-4]
        outfolder = "/".join(outfilepath.split("/")[:-1])
        outsamplename = outfilepath.split("/")[-1][:-4]
        feature_file_name = join(outfolder, outsamplename+".features.npz")
        if args.verbose: 
            print("only working on a file:", outfolder, outsamplename)
        # Process if file does not exists or reprocess flag is set
        
        if (not isfile(feature_file_name)) or args.reprocess:
            umqa.process((join(infolder, insamplename+".pdb"),
                                feature_file_name,
                                args.verbose))
            
        if isfile(feature_file_name):

            model = umqa.myDeepUMQA(twobody_size = 33)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")

            model.load_state_dict(torch.load(join(modelpath, "best.pkl"), map_location=device)['model_state_dict'], weights_only=False)
            model.to(device)
            model.eval()
            
            # Actual prediction
            with torch.no_grad():
                if args.verbose: print("Predicting for", outsamplename) 
                (idx, val), (f1d, bert), f2d, dmy = umqa.getData(feature_file_name)
                f1d = torch.Tensor(f1d).to(device)
                f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
                idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                val = torch.Tensor(val).to(device)

                deviation, mask, lddt, dmy = model(idx, val, f1d, f2d)
                #deviation, mask, lddt, dmy = model(f1d, f2d)
                lddt_np = lddt.cpu().detach().numpy().astype(np.float16)
                if args.per_res_only:
                    np.savez_compressed(outsamplename+".npz",
                            lddt = lddt_np)
                else:
                    np.savez_compressed(outsamplename+".npz",
                            lddt = lddt_np,
                            deviation = deviation.cpu().detach().numpy().astype(np.float16),
                            mask = mask.cpu().detach().numpy().astype(np.float16))

            if not args.leaveTempFile:
                umqa.clean([outsamplename],
                          outfolder,
                          verbose=args.verbose,
                          ensemble=False)
        else:
            print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)
            
            
if __name__== "__main__":
    main()
