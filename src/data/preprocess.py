"""
Preprocessing utilities for Cats vs Dogs: 224x224 RGB, normalize, train/val/test split.
"""
from pathlib import Path
import random
from typing import Tuple

import numpy as np
from PIL import Image


# Standard image size for CNNs
IMG_SIZE = (224, 224)
# ImageNet normalization (optional; use (0, 1) for simpler baseline)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_and_resize(
    image_path: str | Path,
    size: Tuple[int, int] = IMG_SIZE,
    ensure_rgb: bool = True,
) -> np.ndarray:
    """
    Load an image from path, resize to size, and return as RGB numpy array.

    Args:
        image_path: Path to image file.
        size: (height, width) target size; default (224, 224).
        ensure_rgb: If True, convert grayscale to RGB (3 channels).

    Returns:
        Array of shape (H, W, 3), dtype uint8, values in [0, 255].
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    img = img.convert("RGB") if ensure_rgb else np.array(img)
    img = np.array(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    # Resize with PIL for consistency
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(pil_img)


def normalize_uint8_to_float(img: np.ndarray) -> np.ndarray:
    """
    Scale pixel values from [0, 255] to [0, 1].

    Args:
        img: Array (H, W, C) or (C, H, W), uint8.

    Returns:
        Float array, same shape, values in [0, 1].
    """
    return img.astype(np.float32) / 255.0


def normalize_imagenet(img: np.ndarray, channel_last: bool = True) -> np.ndarray:
    """
    Normalize with ImageNet mean and std.
    Expects float in [0, 1] or uint8 [0, 255]; outputs normalized float.

    Args:
        img: (H, W, 3) if channel_last else (3, H, W).
        channel_last: If True, shape is (H, W, 3).

    Returns:
        Normalized float array, same shape.
    """
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    if channel_last:
        img = (img - mean) / std
    else:
        mean = mean.reshape(3, 1, 1)
        std = std.reshape(3, 1, 1)
        img = (img - mean) / std
    return img.astype(np.float32)


def collect_image_paths_and_labels(
    root: str | Path,
    class_names: Tuple[str, ...] = ("cat", "dog"),
) -> list[Tuple[str, int]]:
    """
    Scan root for folders named by class (e.g. cat, dog) and collect (path, label_index).

    Expected layout: root/cat/*.jpg, root/dog/*.jpg (or root/train/cat, root/train/dog).

    Returns:
        List of (absolute_path, label_index). label_index: 0 = cat, 1 = dog.
    """
    root = Path(root)
    pairs: list[Tuple[str, int]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for label_idx, name in enumerate(class_names):
        # Try exact name, then capitalized (e.g. Cat, Dog), then lowercase
        candidates = [
            root / name,
            root / name.capitalize(),
            root / name.lower(),
            root / "train" / name,
            root / "train" / name.capitalize(),
        ]
        folder = None
        for cand in candidates:
            if cand.is_dir():
                folder = cand
                break
        if folder is None:
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in exts:
                pairs.append((str(f.resolve()), label_idx))
    return pairs


def train_val_test_split(
    path_label_pairs: list[Tuple[str, int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list[Tuple[str, int]], list[Tuple[str, int]], list[Tuple[str, int]]]:
    """
    Split (path, label) pairs into train, val, test.

    Args:
        path_label_pairs: List of (path, label).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        (train_list, val_list, test_list), each list of (path, label).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    pairs = list(path_label_pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train_list = pairs[:t]
    val_list = pairs[t : t + v]
    test_list = pairs[t + v :]
    return train_list, val_list, test_list
