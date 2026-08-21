"""Domain losses (Eq. 1–3, Sec. 3.3.3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def boundary_l1(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def depth_l1(pred_depth: Tensor, target_depth: Tensor) -> Tensor:
    return F.l1_loss(pred_depth, target_depth)


def normal_loss(pred: Tensor, target: Tensor) -> Tensor:
    """LGT-Net style normal consistency stub."""
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    return F.l1_loss(dp, dt)


def gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    return normal_loss(pred, target)


def pano_loss(
    pred_b: Tensor,
    tgt_b: Tensor,
    pred_d: Tensor,
    tgt_d: Tensor,
    *,
    lambda_b: float = 1.0,
    mu_d: float = 0.1,
    gamma_g: float = 0.01,
) -> Tensor:
    """Eq. 1: L_pano."""
    lb = boundary_l1(pred_b, tgt_b)
    ld = depth_l1(pred_d, tgt_d)
    lg = normal_loss(pred_b, tgt_b) + gradient_loss(pred_b, tgt_b)
    return lambda_b * lb + mu_d * ld + gamma_g * lg


def perspective_loss(pred_b: Tensor, tgt_b: Tensor, *, delta: float = 1.0) -> Tensor:
    """Eq. 2: L_pp (boundary only)."""
    return delta * boundary_l1(pred_b, tgt_b)


def total_loss(lpano: Tensor, lpp: Tensor) -> Tensor:
    """Eq. 3."""
    return lpano + lpp
