"""Synthetic volumetric patches for CPU stub training."""

from __future__ import annotations

import torch
from torch import Tensor


def synthetic_volume(
    *,
    batch: int = 1,
    channels: int = 1,
    size: int = 32,
    num_classes: int = 4,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, channels, size, size, size, generator=gen)
    # Simple concentric class regions
    c = size // 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), torch.arange(size), indexing="ij"
    )
    r = torch.sqrt((zz - c).float() ** 2 + (yy - c).float() ** 2 + (xx - c).float() ** 2)
    y = torch.zeros(batch, size, size, size, dtype=torch.long)
    m1 = r < size * 0.15
    m2 = (r >= size * 0.15) & (r < size * 0.30)
    m3 = (r >= size * 0.30) & (r < size * 0.45)
    y[:, m1] = 1
    y[:, m2] = 2
    y[:, m3] = 3
    return x, y.clamp(0, num_classes - 1)
