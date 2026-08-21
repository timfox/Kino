"""Auto-exposure loss and bracket EV helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def bracket_evs(*, n: int = 5, span: float = 2.0) -> list[float]:
    """Evenly spaced EV offsets from ``-span`` to ``+span``."""
    if n <= 1:
        return [0.0]
    step = (2.0 * span) / (n - 1)
    return [round(-span + i * step, 4) for i in range(n)]


def ae_loss(frame: Tensor, *, target_mean: float = 0.45, clip_penalty: float = 2.0) -> Tensor:
    """Balance shadow SNR vs highlight clipping for capture planning."""
    lum = frame.mean()
    clip_hi = (frame > 0.98).float().mean()
    clip_lo = (frame < 0.02).float().mean()
    return (lum - target_mean).abs() + clip_penalty * (clip_hi + clip_lo)
