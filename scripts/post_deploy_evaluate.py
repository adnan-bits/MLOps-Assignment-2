#!/usr/bin/env python3
"""
M5: Post-deployment model performance tracking.
Collect a batch of requests (image + true label), call the deployed /predict API,
compute accuracy and optional per-class metrics; write a report.

Usage:
  python scripts/post_deploy_evaluate.py --base-url http://localhost:8000 --csv data/processed/test.csv --limit 50 --out report.json

CSV format: path,label (label 0=cat, 1=dog).
"""
import argparse
import json
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("cat", "dog")


def main() -> int:
    p = argparse.ArgumentParser(description="Post-deploy evaluation: call API and compute accuracy")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "processed" / "test.csv")
    p.add_argument("--limit", type=int, default=50, help="Max number of samples to evaluate")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "post_deploy_report.json")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    # Load (path, label) pairs
    pairs = []
    with open(args.csv) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(",", 1)
            path, label = path.strip(), int(label)
            if Path(path).exists():
                pairs.append((path, label))
            if len(pairs) >= args.limit:
                break

    if not pairs:
        print("No valid paths found in CSV", file=sys.stderr)
        return 1

    url = f"{args.base_url.rstrip('/')}/predict"
    pred_labels = []
    true_labels = []
    errors = 0
    for path, true_label in pairs:
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    url,
                    files={"file": (Path(path).name, f, "image/jpeg")},
                    timeout=30,
                )
            r.raise_for_status()
            data = r.json()
            pred_name = data.get("label", "")
            pred_idx = CLASS_NAMES.index(pred_name) if pred_name in CLASS_NAMES else -1
            if pred_idx < 0:
                errors += 1
                continue
            pred_labels.append(pred_idx)
            true_labels.append(true_label)
        except (requests.RequestException, json.JSONDecodeError, OSError) as e:
            print(f"Error on {path}: {e}", file=sys.stderr)
            errors += 1

    if not pred_labels:
        print("No successful predictions", file=sys.stderr)
        return 1

    # Accuracy
    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = correct / len(pred_labels)

    # Per-class precision/recall (binary: 0=cat, 1=dog)
    tp_cat = sum(1 for p, t in zip(pred_labels, true_labels) if p == 0 and t == 0)
    tp_dog = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 1)
    pred_cat = sum(1 for p in pred_labels if p == 0)
    pred_dog = sum(1 for p in pred_labels if p == 1)
    true_cat = sum(1 for t in true_labels if t == 0)
    true_dog = sum(1 for t in true_labels if t == 1)
    precision_cat = tp_cat / pred_cat if pred_cat else 0.0
    precision_dog = tp_dog / pred_dog if pred_dog else 0.0
    recall_cat = tp_cat / true_cat if true_cat else 0.0
    recall_dog = tp_dog / true_dog if true_dog else 0.0

    report = {
        "n_total": len(pairs),
        "n_success": len(pred_labels),
        "n_errors": errors,
        "accuracy": round(accuracy, 4),
        "precision": {"cat": round(precision_cat, 4), "dog": round(precision_dog, 4)},
        "recall": {"cat": round(recall_cat, 4), "dog": round(recall_dog, 4)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"Accuracy: {accuracy:.4f} (n={len(pred_labels)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
