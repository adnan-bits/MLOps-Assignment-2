"""
Inference utilities: load model, preprocess image, predict class/probabilities.
Used by the training script and the FastAPI service.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.preprocess import IMG_SIZE, load_and_resize, normalize_uint8_to_float
from src.model.cnn import get_model

CLASS_NAMES = ("cat", "dog")


def load_model(path: str | Path, device: str | torch.device | None = None) -> torch.nn.Module:
    """Load BaselineCNN from a .pt checkpoint (state_dict or full model)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = get_model()
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess a single image for the model: 224x224 RGB, [0,1], CHW, batch dim.

    Args:
        image: (H, W, 3) uint8 [0,255] or float [0,1]. Will be resized if not 224x224.

    Returns:
        Tensor (1, 3, 224, 224) float32 on CPU.
    """
    if image.shape[:2] != IMG_SIZE:
        from PIL import Image
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((IMG_SIZE[1], IMG_SIZE[0]))
        image = np.array(pil_img)
    if image.dtype == np.uint8:
        img = normalize_uint8_to_float(image)
    else:
        img = np.clip(image.astype(np.float32), 0.0, 1.0)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def predict_proba(model: torch.nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Return class probabilities (batch, num_classes)."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Run prediction on one image. Returns label name and probabilities.

    Args:
        model: Loaded BaselineCNN.
        image: (H, W, 3) numpy array (uint8 or float).
        device: Optional device for model input.

    Returns:
        {"label": "cat"|"dog", "probabilities": {"cat": float, "dog": float}}
    """
    if device is None:
        device = next(model.parameters()).device
    x = preprocess_image(image).to(device)
    probs = predict_proba(model, x)[0]
    pred_idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[pred_idx],
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }
