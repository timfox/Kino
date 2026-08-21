"""Intensity histograms (Eq. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def channel_histogram(image: Tensor, *, bins: int = 256, eps: float = 1e-12) -> Tensor:
    """Normalized per-channel histogram for ``(C,H,W)`` or ``(H,W)`` in [0,1]."""
    if image.dim() == 2:
        image = image.unsqueeze(0)
    outs = []
    for c in range(image.shape[0]):
        ch = (image[c].clamp(0, 1) * (bins - 1)).long().flatten()
        hist = torch.bincount(ch, minlength=bins).float()
        outs.append(hist / (hist.sum() + eps))
    return torch.stack(outs)


def baseline_histogram(images: list[Tensor], *, bins: int = 256) -> Tensor:
    """Session-0 baseline q_c (Eq. 1)."""
    if not images:
        return torch.ones(3, bins) / bins
    acc = torch.zeros(3, bins)
    for img in images:
        acc += channel_histogram(img, bins=bins)
    return acc / len(images)
