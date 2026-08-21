"""AtomicMotion training losses (Sec. 3.5, Eq. 11)."""

from __future__ import annotations

import torch
from torch import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - target))


def pose_losses(
    pred_rot: Tensor,
    gt_rot: Tensor,
    pred_pos: Tensor,
    gt_pos: Tensor,
    *,
    pred_vel: Tensor | None = None,
    gt_vel: Tensor | None = None,
    pred_acc: Tensor | None = None,
    gt_acc: Tensor | None = None,
) -> dict[str, Tensor]:
    """L_rot, L_pos, L_vel, L_acc (Sec. 3.5)."""
    out = {
        "l_rot": l1_loss(pred_rot, gt_rot),
        "l_pos": l1_loss(pred_pos, gt_pos),
    }
    if pred_vel is not None and gt_vel is not None:
        out["l_vel"] = l1_loss(pred_vel, gt_vel)
    if pred_acc is not None and gt_acc is not None:
        out["l_acc"] = l1_loss(pred_acc, gt_acc)
    out["loss"] = sum(out.values())
    return out


def shape_consistency_loss(beta_seq: Tensor) -> Tensor:
    """L_consist = (1/T) Σ ||β_t - β̄||_1 (Eq. 11)."""
    beta_mean = beta_seq.mean(dim=0, keepdim=True)
    return torch.mean(torch.abs(beta_seq - beta_mean))


def shape_regularization(beta: Tensor) -> Tensor:
    return torch.mean(beta**2)
