import os
import glob
import re
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

# =======================
# 경로 설정
# =======================
ROOT = "/data7/yeonsungkim/DeepUMQA-main"
DECOY_ROOT = os.path.join(ROOT, "data/decoys_casp")

# native pdb 파일 이름 규칙
NATIVE_KEYWORD = "native"

parser = PDBParser(QUIET=True)

# =======================
# CASP 라운드 판별 유틸
# =======================
def get_casp_round(target_id: str) -> str:
    """
    target_id: 예) 'T0950', 'T0953s1', 'T1024', 'T1068s2' 등
    CASP13: T0950 ~ T1016
    CASP14: T1024 ~ T1099
    그 외는 'Other' 로 라벨링
    """
    m = re.match(r"T(\d{4})", target_id)
    if not m:
        return "Other"
    num = int(m.group(1))

    # T0950 ~ T1016
    if 950 <= num <= 1016:
        return "CASP13"

    # T1024 ~ T1099
    if 1024 <= num <= 1099:
        return "CASP14"

    return "Other"

# =======================
# PDB 유틸 (Cβ + Gly는 Cα)
# =======================
def get_cb_like_coords(structure):
    """
    structure에서 첫 번째 모델의 모든 residue에 대해
    - 일반 residue: Cβ 좌표
    - Gly(GLY): Cα 좌표
    를 사용해서 (L, 3) 배열을 반환.

    Cβ가 없고 Cα만 있는 경우 Cα를 fallback으로 사용.
    HETATM(물, 리간드 등)은 제외.
    """
    model = next(structure.get_models())
    coords = []
    for chain in model:
        for residue in chain:
            # residue.id[0] == ' ' 인 것만 표준 아미노산(ATOM)으로 취급
            if residue.id[0] != " ":
                continue

            atom = None
            resname = residue.get_resname().upper()

            if resname == "GLY":
                # 글리신은 Cβ가 없으므로 Cα 사용
                if "CA" in residue:
                    atom = residue["CA"]
            else:
                # 일반 residue는 Cβ 우선, 없으면 Cα fallback
                if "CB" in residue:
                    atom = residue["CB"]
                elif "CA" in residue:
                    atom = residue["CA"]

            if atom is not None:
                coords.append(atom.coord)

    return np.array(coords, dtype=np.float32)


def compute_lddt_cb(native_pdb, model_pdb,
                    cutoff=15.0,
                    thresholds=(0.5, 1.0, 2.0, 4.0),
                    min_seq_sep=1):
    """
    native_pdb와 model_pdb 사이의
    - global Cβ(단, Gly는 Cα) 기반 lDDT
    - residue별 local lDDT 벡터
    를 계산해서 반환.

    (Cβ(+Gly Cα)-only lDDT 버전)
    """
    s_native = parser.get_structure("nat", native_pdb)
    s_model  = parser.get_structure("mod", model_pdb)

    nat_cb = get_cb_like_coords(s_native)  # (L1, 3)
    mod_cb = get_cb_like_coords(s_model)   # (L2, 3)

    L = min(len(nat_cb), len(mod_cb))
    if L == 0:
        return np.nan, np.array([])

    nat_cb = nat_cb[:L]
    mod_cb = mod_cb[:L]

    # pairwise 거리 행렬 (L x L)
    diff_nat = nat_cb[:, None, :] - nat_cb[None, :, :]
    d_nat = np.linalg.norm(diff_nat, axis=-1)  # (L, L)

    diff_mod = mod_cb[:, None, :] - mod_cb[None, :, :]
    d_mod = np.linalg.norm(diff_mod, axis=-1)  # (L, L)

    # sequence index 가정: 0..L-1
    idx = np.arange(L)
    seq_sep = np.abs(idx[:, None] - idx[None, :])  # |i - j|

    # neighbor 마스크:
    #  - native에서 cutoff 이내
    #  - i != j
    #  - seq_sep >= min_seq_sep (너무 가까운 순서 이웃 제외하고 싶으면 2,3 등으로 조정 가능)
    neighbor_mask = (
        (d_nat <= cutoff) &
        (seq_sep >= min_seq_sep) &
        (np.eye(L, dtype=bool) == 0)
    )

    local_lddt = np.full(L, np.nan, dtype=np.float32)

    for i in range(L):
        idx_nei = np.where(neighbor_mask[i])[0]
        if len(idx_nei) == 0:
            continue

        dn = d_nat[i, idx_nei]
        dm = d_mod[i, idx_nei]
        dd = np.abs(dm - dn)  # (num_neighbors,)

        scores_per_thr = []
        for thr in thresholds:
            scores_per_thr.append((dd < thr).astype(np.float32))
        # (num_thresholds, num_neighbors)
        scores_per_thr = np.stack(scores_per_thr, axis=0)
        # threshold 평균 → pair score, 그 다음 neighbor 평균 → residue i lDDT
        per_pair_scores = scores_per_thr.mean(axis=0)
        local_lddt[i] = per_pair_scores.mean()

    if np.all(np.isnan(local_lddt)):
        global_lddt = np.nan
    else:
        global_lddt = np.nanmean(local_lddt)

    return global_lddt, local_lddt  # scalar, (L,)


def corr_safe(x, y, method="pearson"):
    """
    NaN을 pair-wise로 제거한 뒤 상관계수 계산.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    return df.iloc[:, 0].corr(df.iloc[:, 1], method=method)

# =======================
# 메인 로직
# =======================
def main():
    all_rows = []

    # local QA용 누적 리스트 (모든 target/decoy/residue)
    local_pred_all = []   # DeepUMQA local lddt 예측
    local_true_all = []   # label local lddt (Cβ(+Gly Cα) 기반)

    # target별 local QA 요약용
    per_target_local = {}

    # decoys_casp/T*/ 폴더 대상으로 (T0950~T10xx, T1099 등 모두 포함)
    target_dirs = sorted(glob.glob(os.path.join(DECOY_ROOT, "T*")))

    for target_dir in target_dirs:
        if not os.path.isdir(target_dir):
            continue

        target = os.path.basename(target_dir)   # 예: T0950, T0953s1, T1024 등
        casp_round = get_casp_round(target)


        # 해당 타깃의 DeepUMQA CSV (global cb-lddt)
        pred_csv = os.path.join(DECOY_ROOT, f"{target}_deepumqa.csv")
        if not os.path.isfile(pred_csv):
            print(f"[WARN] DeepUMQA csv not found for {target}: {pred_csv}")
            continue

        # native pdb 찾기
        pdb_files = glob.glob(os.path.join(target_dir, "*.pdb"))
        native_candidates = [
            p for p in pdb_files
            if NATIVE_KEYWORD.lower() in os.path.basename(p).lower()
        ]

        if len(native_candidates) == 0:
            print(f"[WARN] No native pdb found for {target} (keyword={NATIVE_KEYWORD})")
            continue
        elif len(native_candidates) > 1:
            print(f"[WARN] Multiple native pdbs found for {target}, using first one:")
            for p in native_candidates:
                print("   ", p)

        native_pdb = native_candidates[0]
        print(f"[INFO] Target {target} (round={casp_round}) / Native: {os.path.basename(native_pdb)}")

        # DeepUMQA global 결과 로드 (structure-level pred_score)
        df_pred = pd.read_csv(pred_csv, sep="\t")   # sample, cb-lddt
        df_pred = df_pred.rename(columns={"sample": "model", "cb-lddt": "pred_score"})

        global_label_list = []
        casp_round_list = [casp_round] * len(df_pred)

        # local QA용 per-target 버퍼
        pt_local_pred = []  # list of arrays
        pt_local_true = []  # list of arrays

        for model_name, pred_score in zip(df_pred["model"], df_pred["pred_score"]):
            model_pdb = os.path.join(target_dir, model_name + ".pdb")
            if not os.path.isfile(model_pdb):
                print(f"[WARN] PDB not found: {model_pdb}")
                global_lddt_label = np.nan
                local_lddt_label = np.array([])
            else:
                # ★ Cβ(+Gly Cα) 기반 lDDT 사용
                global_lddt_label, local_lddt_label = compute_lddt_cb(native_pdb, model_pdb)

            global_label_list.append(global_lddt_label)

            # ==== local QA 부분: pred(local) vs label(local) ====
            npz_path = os.path.join(target_dir, model_name + ".npz")
            if os.path.isfile(npz_path) and local_lddt_label.size > 0:
                data = np.load(npz_path)
                if "lddt" in data:
                    lddt_pred = data["lddt"]
                    # 길이 안 맞을 수 있으니 최소 길이에 맞춰서 자르기
                    L = min(len(lddt_pred), len(local_lddt_label))
                    if L > 0:
                        local_pred = lddt_pred[:L]
                        local_true = local_lddt_label[:L]

                        local_pred_all.append(local_pred)
                        local_true_all.append(local_true)

                        pt_local_pred.append(local_pred)
                        pt_local_true.append(local_true)
                else:
                    print(f"[WARN] 'lddt' not found in {npz_path}")
            else:
                # npz 없으면 local QA는 스킵
                pass

        df_pred["global_lddt_label"] = global_label_list
        df_pred["target"] = target
        df_pred["casp_round"] = casp_round

        all_rows.append(df_pred)

        # target별 local QA summary용 저장
        if pt_local_pred and pt_local_true:
            per_target_local[target] = {
                "pred": pt_local_pred,
                "true": pt_local_true,
                "casp_round": casp_round,
            }

    if not all_rows:
        print("No data collected. Check paths and native keyword / csv / npz.")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    merged_clean = merged.dropna(subset=["pred_score", "global_lddt_label"])

    print("총 유효 샘플 수( global QA ):", len(merged_clean))

    # =======================
    # Global QA correlation (전체)
    # =======================
    global_pearson = corr_safe(
        merged_clean["pred_score"], merged_clean["global_lddt_label"], "pearson"
    )
    global_spearman = corr_safe(
        merged_clean["pred_score"], merged_clean["global_lddt_label"], "spearman"
    )

    print("\n==== Global QA correlation (ALL, DeepUMQA cb-lddt vs Cβ-lDDT label) ====")
    print(f"Pearson  : {global_pearson:.4f}")
    print(f"Spearman : {global_spearman:.4f}")

    # =======================
    # CASP 라운드별 Global QA correlation
    # =======================
    print("\n==== Global QA correlation by CASP round ====")
    for round_name in ["CASP13", "CASP14", "Other"]:
        sub = merged_clean[merged_clean["casp_round"] == round_name]
        if len(sub) == 0:
            continue
        p = corr_safe(sub["pred_score"], sub["global_lddt_label"], "pearson")
        s = corr_safe(sub["pred_score"], sub["global_lddt_label"], "spearman")
        print(f"[{round_name}] N={len(sub)} | Pearson={p:.4f} | Spearman={s:.4f}")

    # =======================
    # Per-target global QA correlation (+ CASP round)
    # =======================
    per_target_global = (
        merged_clean
        .groupby("target", group_keys=False)
        .apply(lambda g: pd.Series({
            "N": len(g),
            "pearson": corr_safe(g["pred_score"], g["global_lddt_label"], "pearson"),
            "spearman": corr_safe(g["pred_score"], g["global_lddt_label"], "spearman"),
            "casp_round": g["casp_round"].iloc[0],
        }))
        .reset_index()
    )
    print("\n==== Per-target Global QA correlation ====")
    print(per_target_global)

    # ===== Global QA correlation (per-target 평균) =====
    global_pearson_mean = np.nanmean(per_target_global["pearson"])
    global_spearman_mean = np.nanmean(per_target_global["spearman"])

    print("\n==== Global QA correlation (per-target average) ====")
    print(f"Global Pearson  (avg over targets): {global_pearson_mean:.4f}")
    print(f"Global Spearman (avg over targets): {global_spearman_mean:.4f}")

    print("\n==== Per-target Global QA correlation ====")
    print(per_target_global)

    # =======================
    # Local QA correlation (residue-level)
    # =======================
    if local_pred_all:
        # 전체
        lp = np.concatenate(local_pred_all)
        lt = np.concatenate(local_true_all)

        local_global_pearson = corr_safe(lp, lt, "pearson")
        local_global_spearman = corr_safe(lp, lt, "spearman")

        print("\n==== Local QA correlation (ALL targets, residue-level, Cβ-lDDT label) ====")
        print(f"Local Pearson  : {local_global_pearson:.4f}")
        print(f"Local Spearman : {local_global_spearman:.4f}")

        # target별 local QA 요약
        rows = []
        for target, d in per_target_local.items():
            pred_flat = np.concatenate(d["pred"])
            true_flat = np.concatenate(d["true"])
            rows.append({
                "target": target,
                "casp_round": d["casp_round"],
                "local_N": len(pred_flat),
                "local_pearson": corr_safe(pred_flat, true_flat, "pearson"),
                "local_spearman": corr_safe(pred_flat, true_flat, "spearman"),
            })
        per_target_local_df = pd.DataFrame(rows)
        # ===== Local QA correlation (per-target 평균) =====
        local_pearson_mean = np.nanmean(per_target_local_df["local_pearson"])
        local_spearman_mean = np.nanmean(per_target_local_df["local_spearman"])

        print("\n==== Local QA correlation (per-target average, residue-level) ====")
        print(f"Local Pearson  (avg over targets): {local_pearson_mean:.4f}")
        print(f"Local Spearman (avg over targets): {local_spearman_mean:.4f}")
        # CASP 라운드별 local QA (모든 residue concat)
        print("\n==== Local QA correlation by CASP round (residue-level) ====")
        for round_name in ["CASP13", "CASP14", "Other"]:
            sub_targets = [t for t, d in per_target_local.items() if d["casp_round"] == round_name]
            if not sub_targets:
                continue
            round_pred = []
            round_true = []
            for t in sub_targets:
                d = per_target_local[t]
                round_pred += d["pred"]
                round_true += d["true"]
            rp = np.concatenate(round_pred)
            rt = np.concatenate(round_true)
            p = corr_safe(rp, rt, "pearson")
            s = corr_safe(rp, rt, "spearman")
            print(f"[{round_name}] local_N={len(rp)} | Pearson={p:.4f} | Spearman={s:.4f}")

        # global + local을 하나의 테이블로 merge
        per_target_all = per_target_global.merge(per_target_local_df, on=["target", "casp_round"], how="left")
    else:
        print("\n[WARN] No local QA data collected (prediction npz/lddt missing?).")
        per_target_all = per_target_global

    # =======================
    # 결과 저장
    # =======================
    out_all = os.path.join(DECOY_ROOT, "deepumqa_vs_lddt_all.csv")
    out_per_target = os.path.join(DECOY_ROOT, "deepumqa_correlation_global_local_per_target_lddt.csv")

    merged_clean.to_csv(out_all, index=False)
    per_target_all.to_csv(out_per_target, index=False)

    print("\n저장 완료:")
    print(f" - {out_all}")
    print(f" - {out_per_target}")

if __name__ == "__main__":
    main()
