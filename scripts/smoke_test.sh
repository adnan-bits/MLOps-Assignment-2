#!/usr/bin/env bash
# Post-deploy smoke test: health and one prediction call. Exit 1 if any check fails.
set -e

BASE_URL="${1:-http://localhost:8000}"
IMAGE_FILE="${2:-}"

echo "Smoke testing $BASE_URL ..."

# Health check
resp=$(curl -sf "$BASE_URL/health")
if ! echo "$resp" | grep -q '"status":"ok"'; then
  echo "Health check failed: $resp"
  exit 1
fi
echo "Health OK"

# Prediction: use provided image or create a minimal test image
if [[ -z "$IMAGE_FILE" || ! -f "$IMAGE_FILE" ]]; then
  IMAGE_FILE="${TMPDIR:-/tmp}/smoke_test_$$.jpg"
  python3 -c "
from PIL import Image
import numpy as np
img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
Image.fromarray(img).save('$IMAGE_FILE')
"
  trap "rm -f $IMAGE_FILE" EXIT
fi

resp=$(curl -sf -X POST -F "file=@$IMAGE_FILE" "$BASE_URL/predict")
if ! echo "$resp" | grep -qE '"label":"(cat|dog)"'; then
  echo "Predict check failed: $resp"
  exit 1
fi
if ! echo "$resp" | grep -q '"probabilities"'; then
  echo "Predict missing probabilities: $resp"
  exit 1
fi
echo "Predict OK"
echo "Smoke tests passed."
