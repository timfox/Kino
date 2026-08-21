"""Haar DWT / IWT for CDAM and WSB (Eq. 1–2, Sec. III-A)."""

from __future__ import annotations

import torch
from torch import Tensor


def haar_dwt2d(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Single-level 2D Haar DWT (even H, W)."""
    if x.dim() != 4:
        raise ValueError(f"expected NCHW, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("H and W must be even for Haar DWT")
    low_rows = (x[:, :, 0::2, :] + x[:, :, 1::2, :]) * 0.5
    high_rows = (x[:, :, 0::2, :] - x[:, :, 1::2, :]) * 0.5
    ll = (low_rows[:, :, :, 0::2] + low_rows[:, :, :, 1::2]) * 0.5
    lh = (low_rows[:, :, :, 0::2] - low_rows[:, :, :, 1::2]) * 0.5
    hl = (high_rows[:, :, :, 0::2] + high_rows[:, :, :, 1::2]) * 0.5
    hh = (high_rows[:, :, :, 0::2] - high_rows[:, :, :, 1::2]) * 0.5
    return ll, lh, hl, hh


def haar_idwt2d(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Inverse Haar (matches :func:`haar_dwt2d`)."""
    b, c, h, w = ll.shape
    low_cols = torch.zeros(b, c, h, w * 2, device=ll.device, dtype=ll.dtype)
    low_cols[..., 0::2] = ll + lh
    low_cols[..., 1::2] = ll - lh
    high_cols = torch.zeros(b, c, h, w * 2, device=ll.device, dtype=ll.dtype)
    high_cols[..., 0::2] = hl + hh
    high_cols[..., 1::2] = hl - hh
    out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, :] = low_cols + high_cols
    out[:, :, 1::2, :] = low_cols - high_cols
    return out
