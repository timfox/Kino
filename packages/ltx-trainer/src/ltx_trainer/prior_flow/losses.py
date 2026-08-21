"""Spherical weighted flow loss (Eq. 17–18)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.prior_flow.config import LOSS_GAMMA
from ltx_trainer.prior_flow.erp_projection import spherical_area_weights


def spherical_l1(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """∥F_pr − F_gt∥_sph^1 (Eq. 17)."""
    diff = (pred - target).abs()
    if diff.dim == 4 and diff.shape[1] == 2:
        diff = diff.sum(dim=1, keepdim=True)
    w = weights / weights.sum().clamp_min(1e-6)
    return (diff * w).sum()


def iterative_flow_loss(
    preds: list[Tensor],
    target: Tensor,
    *,
    gamma: float = LOSS_GAMMA,
) -> Tensor:
    """RAFT-style exponentially weighted sequence loss (Eq. 18)."""
    h, w = target.shape[-2:]
    weights = spherical_area_weights(h, w, device=target.device)
    n = len(preds)
    total = torch.tensor(0.0, device=target.device)
    for i, p in enumerate(preds):
        w_i = gamma ** (n - 1 - i)
        total = total + w_i * spherical_l1(p, target, weights)
    return total
