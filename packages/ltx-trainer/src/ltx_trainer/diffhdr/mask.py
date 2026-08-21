"""Exposure mask detection with temporal EMA (Sec. 3.4, Eq. 6)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.diffhdr.config import MASK_EMA_ALPHA, TAU_HIGH, TAU_LOW


def rec709_luma(rgb: Tensor) -> Tensor:
    if rgb.dim() == 4:
        return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
    return 0.2126 * rgb[0:1] + 0.7152 * rgb[1:2] + 0.0722 * rgb[2:3]


def detect_exposure_masks(
    srgb: Tensor,
    *,
    tau_high: float = TAU_HIGH,
    tau_low: float = TAU_LOW,
) -> tuple[Tensor, Tensor]:
    """Return overexposed and underexposed binary masks [B,1,H,W] or [1,H,W]."""
    y = rec709_luma(srgb)
    over = (y > tau_high).float()
    under = (y < tau_low).float()
    return over, under


def ema_smooth_masks(
    masks: list[Tensor],
    *,
    alpha: float = MASK_EMA_ALPHA,
) -> list[Tensor]:
    """Per-pixel EMA across time: M̃_t = α M_t + (1-α) M̃_{t-1}."""
    if not masks:
        return []
    smoothed: list[Tensor] = [masks[0]]
    for t in range(1, len(masks)):
        prev = smoothed[-1]
        smoothed.append(alpha * masks[t] + (1.0 - alpha) * prev)
    return smoothed
