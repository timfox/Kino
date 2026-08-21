"""Azimuth rotation augmentation (Sec. 4.3, Fig. 6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def azimuth_roll_erp(img: Tensor, theta: float) -> Tensor:
    """Horizontal roll on ERP (longitude shift)."""
    _, _, _, w = img.shape
    shift = int(round(theta / (2 * 3.14159265) * w)) % w
    return torch.roll(img, shifts=shift, dims=-1)


def random_azimuth_augment(img: Tensor, depth: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    theta = torch.rand(1).item() * 2 * 3.14159265
    out_img = azimuth_roll_erp(img, theta)
    out_depth = azimuth_roll_erp(depth, theta) if depth is not None else None
    return out_img, out_depth
