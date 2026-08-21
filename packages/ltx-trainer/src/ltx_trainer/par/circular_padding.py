"""Dual-space circular padding (Eq. 9)."""

from __future__ import annotations

import torch
from torch import Tensor


def circular_pad_width(x: Tensor, ratio: float) -> Tensor:
    """
    Cr(x) = concat(x[..., -rW/2:], x, x[..., :rW/2]) along width.
    x: [..., H, W]
    """
    if ratio <= 0:
        return x
    w = x.shape[-1]
    pad = max(1, int(ratio * w / 2))
    left = x[..., -pad:]
    right = x[..., :pad]
    return torch.cat([left, x, right], dim=-1)


def circular_unpad_width(x: Tensor, ratio: float) -> Tensor:
    """Remove padding regions after VAE decode."""
    if ratio <= 0:
        return x
    w = x.shape[-1]
    inner = int(w / (1 + ratio))
    pad = (w - inner) // 2
    return x[..., pad : pad + inner]


def cyclic_translate_width(x: Tensor, shift: int) -> Tensor:
    """Horizontal cyclic shift Tv for ERP consistency (Sec. 3.3)."""
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)
