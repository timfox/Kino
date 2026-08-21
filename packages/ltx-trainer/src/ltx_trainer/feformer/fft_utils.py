"""3D FFT utilities for frequency-domain modules."""

from __future__ import annotations

import torch
from torch import Tensor


def fft3(x: Tensor) -> Tensor:
    return torch.fft.fftn(x, dim=(-3, -2, -1))


def ifft3(x: Tensor) -> Tensor:
    return torch.fft.ifftn(x, dim=(-3, -2, -1)).real


def frequency_band_decompose(x_freq: Tensor, bands: int = 3) -> list[Tensor]:
    """Split magnitude spectrum into low/mid/high radial bands (Eq. 10)."""
    mag = x_freq.abs()
    d, h, w = mag.shape[-3:]
    cz, cy, cx = d // 2, h // 2, w // 2
    zz = torch.arange(d, device=mag.device, dtype=torch.float32) - cz
    yy = torch.arange(h, device=mag.device, dtype=torch.float32) - cy
    xx = torch.arange(w, device=mag.device, dtype=torch.float32) - cx
    zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing="ij")
    r = torch.sqrt(zz**2 + yy**2 + xx**2)
    r_max = r.max().clamp(min=1.0)
    masks = [
        r <= r_max / 3,
        (r > r_max / 3) & (r <= 2 * r_max / 3),
        r > 2 * r_max / 3,
    ]
    mag = x_freq.abs()
    return [mag * m for m in masks[:bands]]