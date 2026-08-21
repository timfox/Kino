"""Correlation utilities for subjective vs objective scores (Sec. IV)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pearson_correlation(x: Tensor, y: Tensor) -> float:
    """Pearson linear correlation (PLCC)."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def spearman_correlation(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation (SRCC) via ranking."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    def rank(t: Tensor) -> Tensor:
        sorted_idx = torch.argsort(t)
        ranks = torch.empty_like(t, dtype=torch.float32)
        ranks[sorted_idx] = torch.arange(len(t), dtype=torch.float32)
        return ranks

    return pearson_correlation(rank(x), rank(y))


def rmse(x: Tensor, y: Tensor) -> float:
    """Root mean squared error between predictions and MOS."""
    return float(torch.sqrt(torch.mean((x.float() - y.float()) ** 2)))


def fisher_z_mean(correlations: list[float]) -> float:
    """Average correlations with Fisher z-transform (paper Sec. IV)."""
    if not correlations:
        return 0.0
    zs = [math.atanh(max(-0.999, min(0.999, r))) for r in correlations]
    return math.tanh(sum(zs) / len(zs))
