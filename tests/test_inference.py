"""Unit tests for model inference utilities (M3: at least one model/inference function)."""
import numpy as np
import pytest
import torch

from src.model.cnn import get_model
from src.model.inference import (
    CLASS_NAMES,
    preprocess_image,
    predict,
    predict_proba,
)


def test_preprocess_image_output_shape_and_dtype():
    """preprocess_image returns (1, 3, 224, 224) float32 tensor."""
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    out = preprocess_image(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_image_resizes_non_224():
    """preprocess_image resizes non-224 input to 224x224."""
    img = np.random.randint(0, 256, (100, 50, 3), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.shape == (1, 3, 224, 224)


def test_predict_proba_shape_and_sum():
    """predict_proba returns (batch, 2) with rows summing to 1."""
    model = get_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    probs = predict_proba(model, x)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_predict_returns_label_and_probabilities():
    """predict returns dict with 'label' in CLASS_NAMES and 'probabilities' summing to 1."""
    model = get_model()
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = predict(model, img)
    assert "label" in result
    assert result["label"] in CLASS_NAMES
    assert "probabilities" in result
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
