#!/usr/bin/env python3
"""
Prepare data: scan data/raw for cat/dog images, split 80/10/10, write CSVs to data/processed.

Run from repo root:
  python scripts/prepare_data.py

Outputs (DVC-tracked):
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
  data/processed/meta.json (split sizes, seed)
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (
    collect_image_paths_and_labels,
    train_val_test_split,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Support common layouts: raw/train/cat, raw/train/dog; raw/cat, raw/dog;
    # or raw/<dataset_folder>/train/cat after Kaggle unzip
    root = RAW_DIR
    train_dir = RAW_DIR / "train"
    if train_dir.exists():
        root = train_dir
    else:
        subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) == 1:
            inner = subdirs[0]
            if (inner / "train").exists():
                root = inner / "train"
            else:
                root = inner
        # else use RAW_DIR (raw/cat, raw/dog)

    pairs = collect_image_paths_and_labels(root, class_names=("cat", "dog"))
    if not pairs:
        print(f"No cat/dog images found under {root}. Expected subdirs: cat/, dog/ or train/cat/, train/dog/", file=sys.stderr)
        sys.exit(1)

    train_list, val_list, test_list = train_val_test_split(
        pairs, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )

    def write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
        with open(path, "w") as f:
            f.write("path,label\n")
            for p, lbl in rows:
                f.write(f"{p},{lbl}\n")

    write_csv(PROCESSED_DIR / "train.csv", train_list)
    write_csv(PROCESSED_DIR / "val.csv", val_list)
    write_csv(PROCESSED_DIR / "test.csv", test_list)

    meta = {
        "train_size": len(train_list),
        "val_size": len(val_list),
        "test_size": len(test_list),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(train_list)} train, {len(val_list)} val, {len(test_list)} test to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
