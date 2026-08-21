"""Horizontal cyclic treatment for ERP (Sec. V-D, Fig. 14)."""

from __future__ import annotations

import torch
from torch import Tensor


def cyclic_shift_erp(frame: Tensor) -> Tensor:
    """Swap left/right halves along width for edge-aware feature extraction."""
    if frame.dim() != 4:
        raise ValueError("frame must be [B,C,H,W]")
    _, _, _, w = frame.shape
    half = w // 2
    left, right = frame[..., :half], frame[..., half:]
    return torch.cat([right, left], dim=-1)
