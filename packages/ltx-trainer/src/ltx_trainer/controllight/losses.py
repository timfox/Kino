"""Flow matching and misalignment-aware weighted FM (Eq. 3, 6)."""

from __future__ import annotations

import torch
from torch import Tensor


def flow_matching_loss(
    v_pred: Tensor,
    v_target: Tensor,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Eq. (3) L_FM = ||v_θ - v*||² with optional per-location weights."""
    sq = (v_pred - v_target).pow(2)
    if weight is None:
        return sq.mean()
    w = weight
    if w.shape != sq.shape:
        w = w.expand_as(sq)
    return (w * sq).sum() / w.sum().clamp(min=1e-8)


def weighted_flow_matching_loss(
    v_pred: Tensor,
    v_target: Tensor,
    f_ws: Tensor,
) -> Tensor:
    """Eq. (6): L_wFM = sum_u fW(u)||v-v*||² / sum_u fW(u)."""
    return flow_matching_loss(v_pred, v_target, weight=f_ws)


def velocity_target(z1: Tensor, z0: Tensor) -> Tensor:
    """Rectified-flow target v* = z1 - z0."""
    return z1 - z0


def interpolate_latent(z0: Tensor, z1: Tensor, t: float) -> Tensor:
    """z_t = (1-t) z0 + t z1."""
    return (1.0 - t) * z0 + t * z1
