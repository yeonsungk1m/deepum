import numpy as np
from pathlib import Path
import random

# 설정
PISCES_TXT = Path("/data7/yeonsungkim/DeepUMQA-main/pisces_parsed_ids.txt")  # 네가 만든/올린 파일
DECOY_ROOT = Path("/data7/yeonsungkim/DeepUMQA-main/data/train/features")    # 각 타깃 폴더 (native/decoy.features.npz)가 있는 상위 폴더
START_TARGET = "1a1x_A"
END_TARGET = "9bli_C"
TRAIN_RATIO = 0.9

random.seed(0)

# 1) pisces_parsed_ids.txt -> target name 리스트 생성 (1a1x_A 형식)
targets = []
with PISCES_TXT.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) < 2:
            continue
        pdb_id, chain_id = cols[0], cols[1]
        pdb_id = pdb_id.lower()
        target = f"{pdb_id}_{chain_id}"
        targets.append(target)

# 중복 제거 + 정렬
targets = sorted(set(targets))

# 2) 정렬 기준으로 START_TARGET ~ END_TARGET 사이에 있는 타깃만 선택
subset_targets = [t for t in targets if START_TARGET <= t <= END_TARGET]

print(f"Subset range [{START_TARGET} ~ {END_TARGET}] -> {len(subset_targets)} targets (before filtering).")

# 3) 실제 feature가 존재하는 타깃만 필터링
usable_targets = []
for t in subset_targets:
    tdir = DECOY_ROOT / t
    if not tdir.is_dir():
        continue

    native_fea = tdir / "native.features.npz"
    decoy_feas = [p for p in tdir.glob("*.features.npz") if p.name != "native.features.npz"]

    # native.features.npz 있고, decoy.features.npz도 최소 1개 이상 있어야만 사용
    if native_fea.exists() and len(decoy_feas) > 0:
        usable_targets.append(t)

print(f"Usable targets with features: {len(usable_targets)}")

if len(usable_targets) == 0:
    raise SystemExit("No usable targets found in the specified range. Check data/decoys structure & features.")

# 4) 셔플 후 8:2 split
random.shuffle(usable_targets)
split_idx = int(len(usable_targets) * TRAIN_RATIO)

train_targets = usable_targets[:split_idx]
valid_targets = usable_targets[split_idx:]

print(f"Train: {len(train_targets)}, Valid: {len(valid_targets)}")

# 5) .npy로 저장
Path("data").mkdir(exist_ok=True)

np.save("data/train_decoys.npy", np.array(train_targets, dtype=object))
np.save("data/valid_decoys.npy", np.array(valid_targets, dtype=object))

print("Saved:")
print("  data/train_decoys.npy")
print("  data/valid_decoys.npy")
