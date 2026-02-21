#!/usr/bin/env python3
"""
Download Cats vs Dogs dataset from Kaggle into data/raw.

Requires:
  - Kaggle API credentials: place kaggle.json in ~/.kaggle/
  - Install: pip install kaggle

Usage:
  python scripts/download_data.py

Dataset: "bhavikajain/cats-and-dogs-images" (or set KAGGLE_DATASET env var).
Output: data/raw/ with train/cat/, train/dog/ (or similar per dataset structure).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# Common Cats vs Dogs datasets on Kaggle
DEFAULT_DATASET = "bhavikajain/cats-and-dogs-images"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = os.environ.get("KAGGLE_DATASET", DEFAULT_DATASET)

    try:
        import kaggle
    except ImportError:
        print("Install kaggle: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading dataset: {dataset}")
    exit_code = subprocess.call(
        ["kaggle", "datasets", "download", "-p", str(RAW_DIR), "--unzip", dataset],
        cwd=str(REPO_ROOT),
    )
    if exit_code != 0:
        print("Kaggle download failed. Ensure kaggle.json is in ~/.kaggle/", file=sys.stderr)
        sys.exit(exit_code)
    print(f"Data written to {RAW_DIR}")


if __name__ == "__main__":
    main()
