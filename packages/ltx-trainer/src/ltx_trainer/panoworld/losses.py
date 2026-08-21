"""Pano-native instruction losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def choice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets)


def bfov_regression_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return pred.sum() * 0.0
    p = pred[mask]
    t = target[mask]
    return F.smooth_l1_loss(p, t)


def total_training_loss(
    choice_logits: Tensor,
    answers: Tensor,
    bfov_pred: Tensor,
    bfov_gt: Tensor,
    bfov_mask: Tensor,
    *,
    bfov_weight: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    l_mc = choice_loss(choice_logits, answers)
    l_bfov = bfov_regression_loss(bfov_pred, bfov_gt, bfov_mask)
    loss = l_mc + bfov_weight * l_bfov
    return loss, {
        "l_mc": float(l_mc.item()),
        "l_bfov": float(l_bfov.item()),
        "loss": float(loss.item()),
    }
