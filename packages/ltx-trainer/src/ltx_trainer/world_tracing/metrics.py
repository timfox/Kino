"""Depth and point-cloud metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def mean_absolute_error(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    diff = (pred - target).abs()
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean().item())


def root_mean_square_error(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(diff.sqrt().mean().item())


def abs_relative_error(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    rel = (pred - target).abs() / target.abs().clamp_min(1e-3)
    if mask is not None:
        rel = rel[mask]
    return float(rel.mean().item())


def delta_accuracy(pred: Tensor, target: Tensor, thresh: float = 1.25, mask: Tensor | None = None) -> float:
    ratio = torch.max(pred / target.clamp_min(1e-3), target / pred.clamp_min(1e-3))
    ok = ratio < thresh
    if mask is not None:
        ok = ok[mask]
    return float(ok.float().mean().item())


def chamfer_l2(pred: Tensor, target: Tensor) -> float:
    """Bidirectional nearest-neighbor L2 (subsampled for smoke)."""
    p = pred.reshape(-1, 3)
    t = target.reshape(-1, 3)
    n = min(256, p.shape[0], t.shape[0])
    p = p[:n]
    t = t[:n]
    d = torch.cdist(p, t)
    return float((d.min(dim=1).values.pow(2).mean() + d.min(dim=0).values.pow(2).mean()).item())
