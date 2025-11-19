#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# ===== 경로 설정 =====

ROOT = Path("/data7/yeonsungkim/DeepUMQA-main")

CASP13_TARGETS = ROOT / "data/casp14/targets"             # T0950.pdb 같은 native
CASP13_SRV_ROOT = ROOT / "data/casp14/server_predictions" # T0950/T0950/... decoys
OUT_ROOT = ROOT / "data/decoys_casp13"                    # 새로 만들 decoy 폴더 루트

OUT_ROOT.mkdir(parents=True, exist_ok=True)                 # 새로 만들 decoy 폴더 루트


def collect_targets():
    """targets 디렉토리에서 Txxxx.pdb 목록을 가져와서 타깃 ID 리스트로 반환."""
    target_ids = []
    for pdb in sorted(CASP13_TARGETS.glob("T*.pdb")):
        tid = pdb.stem  # 예: T0950.pdb -> T0950
        target_ids.append(tid)
    return target_ids


def find_decoy_files_for_target(tid: str):
    """
    server_predictions 안에서 해당 타깃(tid)에 대한 decoy 파일들을 전부 찾는다.
    예:
      /.../server_predictions/T0950/T0950/3D-JIGSAW_SL1_TS1/...
    처럼 한 번 더 T0950 디렉토리가 중첩된 경우도 있으니 재귀적으로 찾는다.
    """
    decoys = []

    # 1) 기본 폴더: server_predictions/T0950
    base_dir = CASP13_SRV_ROOT / tid
    if not base_dir.is_dir():
        print(f"[WARN] server_predictions for {tid} not found: {base_dir}")
        return decoys

    # 2) 어떤 tar를 풀었느냐에 따라 구조가 다를 수 있으니,
    #    재귀적으로 파일(.pdb or 확장자 없는 것)을 전부 모으자.
    for path in base_dir.rglob("*"):
        if path.is_file():
            # tar.gz 파일 같은 건 제외
            if path.suffix in [".tar", ".gz"]:
                continue
            # CASP TS 파일이 확장자 없이 있는 경우도 있으므로 일단 모두 포함.
            decoys.append(path)

    return sorted(decoys)


def main():
    target_ids = collect_targets()
    print(f"[INFO] Found {len(target_ids)} targets in {CASP13_TARGETS}")

    for tid in target_ids:
        native_pdb = CASP13_TARGETS / f"{tid}.pdb"
        if not native_pdb.exists():
            print(f"[WARN] native pdb missing for {tid}: {native_pdb}")
            continue

        out_dir = OUT_ROOT / tid
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) native 복사
        out_native = out_dir / "native.pdb"
        if not out_native.exists():
            shutil.copyfile(native_pdb, out_native)
            print(f"[OK] {tid}: native.pdb 복사 완료")
        else:
            print(f"[SKIP] {tid}: native.pdb 이미 존재")

        # 2) decoy 수집
        decoy_files = find_decoy_files_for_target(tid)
        if not decoy_files:
            print(f"[WARN] {tid}: decoy 파일을 찾지 못함 (server_predictions 쪽 구조 확인 필요)")
            continue

        # 3) decoy 파일들을 T0950_01.pdb, T0950_02.pdb, ... 형식으로 복사
        cnt = 0
        for src in decoy_files:
            cnt += 1
            out_name = f"{tid}_{cnt:03d}.pdb"
            dst = out_dir / out_name

            if dst.exists():
                # 이미 복사된 경우(재실행 등)는 스킵
                continue

            shutil.copyfile(src, dst)

        print(f"[OK] {tid}: decoy {cnt}개 복사 완료 -> {out_dir}")

    print("[DONE] CASP13 native + decoy merge 완료")


if __name__ == "__main__":
    main()
