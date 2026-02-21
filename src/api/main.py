"""
FastAPI inference service for Cats vs Dogs: health check and prediction.
M5: Request/response logging (no sensitive data), basic request count and latency metrics.
"""
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from PIL import Image

from src.model.inference import load_model, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model path (override with MODEL_PATH env var)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "baseline.pt"

_model = None

# In-app metrics (M5): request count and total latency for /predict
_request_count = 0
_total_latency_ms = 0.0


def get_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global _model
    path = get_model_path()
    if path.exists():
        _model = load_model(path)
    else:
        _model = None
    yield
    _model = None


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, latency. No request/response bodies or image data."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        # Track metrics for /predict only (successful responses)
        if request.url.path == "/predict" and response.status_code == 200:
            global _request_count, _total_latency_ms  # noqa: PLW0603
            _request_count += 1
            _total_latency_ms += latency_ms
        return response


app = FastAPI(title="Cats vs Dogs API", description="Binary image classification", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


@app.get("/health")
def health():
    """Health check: returns ok and whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.get("/metrics")
def metrics():
    """Basic metrics: request count and average latency for /predict (from in-app counters)."""
    avg_ms = _total_latency_ms / _request_count if _request_count else 0.0
    return {
        "request_count": _request_count,
        "avg_latency_ms": round(avg_ms, 2),
    }


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict class (cat or dog) and probabilities for an uploaded image.
    Accepts: image file (e.g. image/jpeg, image/png).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.convert("RGB")
        arr = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    result = predict(_model, arr)
    return result
