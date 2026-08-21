"""Norm-in-norm loss (Eq. 22, Li et al. 2020)."""

from __future__ import annotations

import torch
from torch import Tensor


def norm_in_norm_loss(
    pred: Tensor,
    target: Tensor,
    *,
    gamma: float = 4.0,
    eps: float = 1e-8,
) -> Tensor:
    """Batch-normalized L_gamma distance between MOS vectors."""
    if pred.numel() == 0:
        return pred.sum()
    pm, tm = pred.mean(), target.mean()
    ps = pred.std(unbiased=False).clamp_min(eps)
    ts = target.std(unbiased=False).clamp_min(eps)
    pn = (pred - pm) / ps
    tn = (target - tm) / ts
    return ((pn - tn).abs() ** gamma).mean()
