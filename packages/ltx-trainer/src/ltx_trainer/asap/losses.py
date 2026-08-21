"""Flow-matching localized loss, preference gap, DPO and margin-bounded objectives (Eq. 3–6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def localized_velocity_mse(
    v_pred: Tensor,
    v_target: Tensor,
    mask: Tensor,
    *,
    alpha: float = 10.0,
) -> Tensor:
    """Eq. (4): ``(1 + αM) ⊙ ||v_θ - (x_1 - x_0)||^2`` averaged over batch/spatial dims.

    ``mask`` is broadcastable to ``v_pred`` (e.g. (B,1,H,W) with 1 on anatomical regions).
    """
    weight = 1.0 + alpha * mask
    sq = (v_pred - v_target).pow(2)
    if sq.shape != weight.shape:
        weight = weight.expand_as(sq)
    return (weight * sq).mean()


def flow_matching_loss(
    v_pred: Tensor,
    v_target: Tensor,
    mask: Tensor | None = None,
    *,
    alpha: float = 0.0,
) -> Tensor:
    """Scalar FM loss; localized when ``mask`` and ``alpha > 0``."""
    if mask is None or alpha <= 0:
        return F.mse_loss(v_pred, v_target)
    return localized_velocity_mse(v_pred, v_target, mask, alpha=alpha)


def preference_gap(
    loss_policy_w: Tensor,
    loss_policy_l: Tensor,
    loss_ref_w: Tensor,
    loss_ref_l: Tensor,
) -> Tensor:
    """Eq. (2) / (5): δ ≈ -(L_w - L_l - L_ref_w - L_ref_l) (scalar or per-batch)."""
    return -(loss_policy_w - loss_policy_l - loss_ref_w - loss_ref_l)


def diffusion_dpo_loss(
    delta: Tensor,
    *,
    beta: float = 5000.0,
) -> Tensor:
    """Eq. (3): ``-log σ(β δ)``."""
    return -F.logsigmoid(beta * delta).mean()


def margin_bounded_loss(
    delta_local: Tensor,
    *,
    tau: float = 0.01,
) -> Tensor:
    """Eq. (6): ``(δ_local - τ)²``."""
    return (delta_local - tau).pow(2).mean()


def anatomical_alignment_loss(
    loss_policy_w: Tensor,
    loss_policy_l: Tensor,
    loss_ref_w: Tensor,
    loss_ref_l: Tensor,
    *,
    tau: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Full ASAP Phase-II objective: bounded margin on localized preference gap.

    Returns:
        total loss, detached ``delta_local`` for logging.
    """
    delta = preference_gap(loss_policy_w, loss_policy_l, loss_ref_w, loss_ref_l)
    return margin_bounded_loss(delta, tau=tau), delta.detach()
