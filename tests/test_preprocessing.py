"""Unit tests for data preprocessing (M3: at least one pre-processing function)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.preprocess import (
    IMG_SIZE,
    load_and_resize,
    normalize_uint8_to_float,
    train_val_test_split,
)


def test_normalize_uint8_to_float_shape_and_range():
    """Normalize [0,255] uint8 to [0,1] float; shape and dtype preserved."""
    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    out = normalize_uint8_to_float(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out[0, 0, 0], img[0, 0, 0] / 255.0)


def test_normalize_uint8_to_float_boundaries():
    """Check 0 -> 0 and 255 -> 1."""
    zero = np.zeros((10, 10, 3), dtype=np.uint8)
    one = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert np.allclose(normalize_uint8_to_float(zero), 0.0)
    assert np.allclose(normalize_uint8_to_float(one), 1.0)


def test_load_and_resize_returns_224x224_rgb(tmp_path):
    """load_and_resize returns (224, 224, 3) uint8 from an image file."""
    # Create a small RGB image file
    img = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)
    path = tmp_path / "test.jpg"
    Image.fromarray(img).save(path)
    out = load_and_resize(path, size=IMG_SIZE, ensure_rgb=True)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_load_and_resize_raises_on_missing_file():
    """load_and_resize raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_and_resize(Path("/nonexistent/image.jpg"))


def test_train_val_test_split_ratios_and_disjoint():
    """Split has correct sizes (80/10/10) and no overlap."""
    pairs = [(f"/path/{i}.jpg", i % 2) for i in range(1000)]
    train, val, test = train_val_test_split(
        pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    test_paths = {p for p, _ in test}
    assert train_paths.isdisjoint(val_paths) and val_paths.isdisjoint(test_paths) and train_paths.isdisjoint(test_paths)
