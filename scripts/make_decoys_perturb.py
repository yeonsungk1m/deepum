#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import shutil
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
# PyRosetta
from pyrosetta import init, pose_from_pdb, MoveMap
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.protocols.simple_moves import SmallMover, ShearMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.moves import MonteCarlo

# ---------------------------
# 공통 유틸
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1) Biopython 기반 all-atom perturbation
# ---------------------------
def random_rigid(delta_t=0.5, delta_deg=3.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    theta = np.deg2rad(rng.uniform(-delta_deg, delta_deg))
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    t = rng.normal(0.0, delta_t, size=3)
    return R, t

class ChainSelect(Select):
    def __init__(self, keep_chain=None, include_hetatm=False, include_water=False):
        self.keep_chain = keep_chain
        self.include_hetatm = include_hetatm
        self.include_water = include_water

    def accept_chain(self, chain):
        # keep_chain이 None이면 모든 체인, 아니면 해당 체인만
        return (self.keep_chain is None) or (chain.id == self.keep_chain)

    def accept_residue(self, residue):
        hf = residue.id[0].strip()  # '' or 'H_' or 'W'
        if hf.startswith('W'):
            return self.include_water      # 물 포함 여부
        if hf != '':
            return self.include_hetatm     # 리간드(HETATM) 포함 여부
        return 1


def perturb_all_atoms(
    native_pdb: Path,
    out_pdb: Path,
    chain_id: str = None,
    delta_t: float = 0.5,
    delta_deg: float = 3.0,
    atom_jitter: float = 0.0,
    include_hetatm: bool = False,
    include_water: bool = False,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(native_pdb.stem, str(native_pdb))
    model = next(struct.get_models())

    chains = [c for c in model.get_chains()
              if (chain_id is None or c.id == chain_id)]

    for chain in chains:
        for res in chain.get_residues():
            hf = res.id[0].strip()
            if hf.startswith('W') and not include_water:
                continue
            if hf != '' and not include_hetatm:
                continue

            atoms = list(res.get_atoms())
            if not atoms:
                continue

            ca = [a for a in atoms if a.get_id() == "CA"]
            center = ca[0].coord if ca else np.mean([a.coord for a in atoms], axis=0)

            R, t = random_rigid(delta_t=delta_t, delta_deg=delta_deg, rng=rng)
            for a in atoms:
                new = (a.coord - center) @ R.T + center + t
                if atom_jitter > 0.0:
                    new = new + rng.normal(0.0, atom_jitter, size=3)
                a.set_coord(new)

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_pdb), ChainSelect(keep_chain=chain_id,
                                      include_hetatm=include_hetatm,
                                      include_water=include_water))

# ---------------------------
# 2) PyRosetta 기반 decoy 생성
# ---------------------------
_PYRO_INIT = False

def init_pyrosetta():
    global _PYRO_INIT
    if _PYRO_INIT:
        return
    # 필요하면 옵션 더 추가 가능
    init("-constant_seed -mute all -read_only_ATOM_entries")
    _PYRO_INIT = True

def make_pyrosetta_decoy(
    in_pdb: Path,
    out_pdb: Path,
    temperature: float = 2.0,
    n_cycles: int = 5,
    moves_per_cycle: int = 200,
):
    """
    - in_pdb: 시작 구조 (native 또는 perturb된 구조)
    - MonteCarlo + small/shear mover로 backbone 살짝씩 흔들고
    - 마지막에 MinMover로 에너지 최소화 후 PDB 저장
    """
    init_pyrosetta()

    pose = pose_from_pdb(str(in_pdb))
    scorefxn = get_score_function()
    kT = temperature

    # 1) backbone/chi 모두 허용하는 MoveMap
    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)

    # 2) MoveMap을 사용하는 small / shear mover
    #   (angle=5deg, nmoves=5는 필요에 따라 조정 가능)
    small = SmallMover(mm, 5.0, 5)
    shear = ShearMover(mm, 5.0, 5)

    # 3) Monte Carlo: scorefxn은 여기만 물려주면 됨
    mc = MonteCarlo(pose, scorefxn, kT)

    for c in range(n_cycles):
        for i in range(moves_per_cycle):
            if i % 2 == 0:
                small.apply(pose)
            else:
                shear.apply(pose)
            # move 적용 후 에너지 평가 + accept/reject
            mc.boltzmann(pose)
        # 각 사이클 끝에서 최소 에너지 pose 복구
        mc.recover_low(pose)

    # 4) 마지막에 최소화
    minm = MinMover()
    minm.movemap(mm)
    minm.score_function(scorefxn)
    minm.min_type("lbfgs_armijo_nonmonotone")
    minm.apply(pose)

    # 5) 결과 저장
    pose.dump_pdb(str(out_pdb))



# ---------------------------
# 3) 타깃 루프
# ---------------------------
def infer_chain_from_target(target: str):
    # 1a1x_A → A, 7pgz_AAA → AAA
    if "_" in target:
        return target.split("_", 1)[1]
    return None

def main():
    ap = argparse.ArgumentParser(description="native + perturb + PyRosetta decoy 생성")
    ap.add_argument("--native_dir", required=True, help="native_chains 디렉토리 (예: data/native_chains)")
    ap.add_argument("--out_root", required=True, help="출력 decoy 루트 (예: data/decoys)")
    ap.add_argument("--targets", nargs="*", default=None, help="처리할 타깃 리스트. 미지정 시 native_dir의 *.pdb 전부")

    # perturb 설정
    ap.add_argument("--delta_t", type=float, default=0.5)
    ap.add_argument("--delta_deg", type=float, default=3.0)
    ap.add_argument("--atom_jitter", type=float, default=0.1)
    ap.add_argument("--include_hetatm", action="store_true")
    ap.add_argument("--include_water", action="store_true")

    # PyRosetta 샘플링 설정 (강한 decoy 기준)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--moves_per_cycle", type=int, default=200)

    args = ap.parse_args()

    native_dir = Path(args.native_dir)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # strong 파라미터
    temp_strong = args.temperature
    cycles_strong = args.cycles
    moves_strong = args.moves_per_cycle

    # mid decoy용 파라미터 (자동 스케일링)
    temp_mid = max(0.5, temp_strong * 0.5)
    cycles_mid = max(1, cycles_strong // 2)
    moves_mid = max(50, moves_strong // 2)

    if args.targets:
        targets = args.targets
    else:
        targets = [p.stem for p in sorted(native_dir.glob("*.pdb"))]

    for target in targets:
        native_pdb = native_dir / f"{target}.pdb"
        if not native_pdb.exists():
            print(f"[WARN] native PDB not found: {native_pdb}")
            continue

        out_dir = out_root / target
        ensure_dir(out_dir)

        # 0) native 복사
        native_out = out_dir / "native.pdb"
        if not native_out.exists():
            shutil.copyfile(native_pdb, native_out)

        chain_id = infer_chain_from_target(target)

        # 1) 약한 perturb decoy → {target}_01.pdb
        perturb_out = out_dir / f"{target}_01.pdb"
        if not perturb_out.exists():
            try:
                perturb_all_atoms(
                    native_pdb=native_pdb,
                    out_pdb=perturb_out,
                    chain_id=chain_id,
                    delta_t=args.delta_t,
                    delta_deg=args.delta_deg,
                    atom_jitter=args.atom_jitter,
                    include_hetatm=args.include_hetatm,
                    include_water=args.include_water,
                    seed=0,
                )
                print(f"[OK] {target}: {perturb_out.name} 생성")
            except Exception as e:
                print(f"[WARN] perturb 실패 {target}: {e}")
                perturb_out = None

        # PyRosetta 시작 구조: 있으면 perturb, 없으면 native
        start_pdb = perturb_out if perturb_out and perturb_out.exists() else native_out

        # 2) 중간 강도 PyRosetta decoy → {target}_02.pdb
        pyro_mid_out = out_dir / f"{target}_02.pdb"
        if not pyro_mid_out.exists():
            try:
                make_pyrosetta_decoy(
                    in_pdb=start_pdb,
                    out_pdb=pyro_mid_out,
                    temperature=temp_mid,
                    n_cycles=cycles_mid,
                    moves_per_cycle=moves_mid,
                )
                print(
                    f"[OK] {target}: {pyro_mid_out.name} 생성 "
                    f"(T={temp_mid}, cycles={cycles_mid}, moves={moves_mid})"
                )
            except Exception as e:
                print(f"[WARN] PyRosetta 중간 decoy 생성 실패 {target}: {e}")

        # 3) 강한 PyRosetta decoy → {target}_03.pdb
        pyro_strong_out = out_dir / f"{target}_03.pdb"
        if not pyro_strong_out.exists():
            try:
                make_pyrosetta_decoy(
                    in_pdb=start_pdb,
                    out_pdb=pyro_strong_out,
                    temperature=temp_strong,
                    n_cycles=cycles_strong,
                    moves_per_cycle=moves_strong,
                )
                print(
                    f"[OK] {target}: {pyro_strong_out.name} 생성 "
                    f"(T={temp_strong}, cycles={cycles_strong}, moves={moves_strong})"
                )
            except Exception as e:
                print(f"[WARN] PyRosetta 강한 decoy 생성 실패 {target}: {e}")


if __name__ == "__main__":
    main()
