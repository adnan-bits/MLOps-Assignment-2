"""
PyTorch Dataset for Cats vs Dogs using data/processed CSV splits.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocess import IMG_SIZE, load_and_resize, normalize_uint8_to_float


class CatsDogsDataset(Dataset):
    """Dataset that loads (path, label) from a CSV and returns tensor (C, H, W), label."""

    def __init__(
        self,
        csv_path: str | Path,
        size: tuple[int, int] = IMG_SIZE,
        augment: bool = False,
    ):
        self.size = size
        self.augment = augment
        self.samples: list[tuple[str, int]] = []
        with open(csv_path) as f:
            next(f)  # header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(",", 1)
                self.samples.append((path.strip(), int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = load_and_resize(path, size=self.size, ensure_rgb=True)
        img = normalize_uint8_to_float(img)
        if self.augment:
            img = _augment(img)
        # HWC -> CHW, float32
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, label


def _augment(img: np.ndarray) -> np.ndarray:
    """Simple augmentation: horizontal flip with 50% prob."""
    if np.random.random() > 0.5:
        img = np.fliplr(img).copy()
    return img
