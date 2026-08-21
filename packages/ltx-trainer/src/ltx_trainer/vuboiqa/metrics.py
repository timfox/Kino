"""SRCC / PLCC correlation stubs (Sec. 4.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def spearman_proxy(pred: Tensor, target: Tensor) -> Tensor:
    """Rank-correlation proxy via Pearson on ranks."""
    rp = pred.argsort().argsort().float()
    rt = target.argsort().argsort().float()
    rp = (rp - rp.mean()) / rp.std().clamp_min(1e-8)
    rt = (rt - rt.mean()) / rt.std().clamp_min(1e-8)
    return (rp * rt).mean()


def pearson(pred: Tensor, target: Tensor) -> Tensor:
    p = (pred - pred.mean()) / pred.std().clamp_min(1e-8)
    t = (target - target.mean()) / target.std().clamp_min(1e-8)
    return (p * t).mean()
