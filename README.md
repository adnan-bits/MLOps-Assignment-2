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

1. Clone and install: `pip install -r requirements.txt` (use a venv).
2. **Data:** Place images in `data/raw` with subdirs `Cat/` and `Dog/` (or `cat/`/`dog/`; `train/cat/`, `train/dog/` also supported). Then run `python scripts/prepare_data.py` to create train/val/test splits in `data/processed/`.
3. **DVC:** `dvc init` (optional). Reproduce pipeline: `dvc repro` (creates `data/processed/` with train/val/test splits).
4. Run API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` (or Docker). **Docker:** `docker build -t cats-dogs-api .` then `docker run -p 8000:8000 -v $(pwd)/models:/app/models cats-dogs-api`. Then `curl http://localhost:8000/health` and `curl -X POST -F "file=@image.jpg" http://localhost:8000/predict`.
5. Run tests: `python -m pytest tests/ -v` (from repo root).

## Assignment modules

- **M1** – Git + DVC, baseline model, MLflow tracking
- **M2** – FastAPI, requirements.txt, Dockerfile
- **M3** – Pytest, CI (GitHub Actions: test → build image → push to GHCR on main)
- **M4** – Deployment (Docker Compose in deployment/), CD job on main + smoke test (scripts/smoke_test.sh)
- **M5** – Logging (request/path/status/latency), `/metrics` (count + avg latency), post-deploy report: `scripts/post_deploy_evaluate.py --csv data/processed/test.csv --limit 50 --out report.json`
