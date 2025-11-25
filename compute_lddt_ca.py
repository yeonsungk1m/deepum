from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser(QUIET=True)

def get_ca_coords(structure):
    model = next(structure.get_models())
    coords = []
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                coords.append(residue["CA"].coord)
    return np.array(coords)  # (L, 3)

def compute_lddt_ca(native_pdb, model_pdb,
                    cutoff=15.0,
                    thresholds=(0.5, 1.0, 2.0, 4.0)):
    s_native = parser.get_structure("nat", native_pdb)
    s_model  = parser.get_structure("mod", model_pdb)

    nat_ca = get_ca_coords(s_native)
    mod_ca = get_ca_coords(s_model)

    L = min(len(nat_ca), len(mod_ca))
    if L == 0:
        return np.nan, np.array([])

    nat_ca = nat_ca[:L]
    mod_ca = mod_ca[:L]

    # pairwise 거리 행렬 (L x L)
    # (실제 구현에선 효율 위해 chunking/벡터화 좀 더 할 수 있음)
    diff = nat_ca[:, None, :] - nat_ca[None, :, :]
    d_nat = np.linalg.norm(diff, axis=-1)  # (L, L)

    diff_m = mod_ca[:, None, :] - mod_ca[None, :, :]
    d_mod = np.linalg.norm(diff_m, axis=-1)  # (L, L)

    # neighbor 마스크: native에서 cutoff 이내이면서 i!=j
    neighbor_mask = (d_nat <= cutoff) & (np.eye(L) == 0)

    local_lddt = np.zeros(L, dtype=np.float32)
    local_lddt[:] = np.nan

    for i in range(L):
        idx = np.where(neighbor_mask[i])[0]
        if len(idx) == 0:
            continue
        dn = d_nat[i, idx]
        dm = d_mod[i, idx]
        dd = np.abs(dm - dn)

        scores = []
        for thr in thresholds:
            scores.append((dd < thr).astype(np.float32))
        # (4, num_neighbors) -> 평균
        scores = np.stack(scores, axis=0).mean(axis=0)  # per-pair score
        local_lddt[i] = scores.mean()                   # residue i lDDT

    # global lDDT: NaN 제외 평균
    if np.all(np.isnan(local_lddt)):
        global_lddt = np.nan
    else:
        global_lddt = np.nanmean(local_lddt)

    return global_lddt, local_lddt
