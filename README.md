# MLOps Assignment 2 – Cats vs Dogs

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Structure

- `src/data` – Preprocessing and dataset loading (224×224 RGB, 80/10/10 split)
- `src/model` – Model definition, training, inference utilities
- `src/api` – FastAPI inference service (health + predict)
- `scripts/` – Training, data download, smoke tests
- `tests/` – Unit tests (pytest)
- `deployment/` – Docker Compose (and optional Kubernetes) manifests
- `data/` – Raw and processed data (DVC-tracked)

## Quick start

**Using conda env `ai` (in project root, with `ai` already activated):**

```bash
# 1. Install deps (if not already)
pip install -r requirements.txt -r requirements-dev.txt

# 2. Run tests
python -m pytest tests/ -v

# 3. Data: download from Kaggle (needs ~/.kaggle/kaggle.json) or put images in data/raw (Cat/, Dog/ or train/cat/, train/dog/)
python scripts/download_data.py
python scripts/prepare_data.py

# 4. Train (writes models/baseline.pt and MLflow runs in mlruns/)
python scripts/train.py --epochs 5

# 5. Start API (from repo root)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# In another terminal: curl http://localhost:8000/health
# curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

**Alternative (venv):** `pip install -r requirements.txt` in a venv, then same steps 2–5.

**Docker:** `docker build -t cats-dogs-api .` then `docker run -p 8000:8000 -v $(pwd)/models:/app/models cats-dogs-api`.

## Assignment modules

- **M1** – Git + DVC, baseline model, MLflow tracking
- **M2** – FastAPI, requirements.txt, Dockerfile
- **M3** – Pytest, CI (GitHub Actions: test → build image → push to GHCR on main)
- **M4** – Deployment (Docker Compose in deployment/), CD job on main + smoke test (scripts/smoke_test.sh)
- **M5** – Logging (request/path/status/latency), `/metrics` (count + avg latency), post-deploy report: `scripts/post_deploy_evaluate.py --csv data/processed/test.csv --limit 50 --out report.json`
