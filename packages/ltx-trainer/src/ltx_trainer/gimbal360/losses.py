"""Gimbal360 training losses (Sec. 3.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def smooth_l1_flow(pred: Tensor, target: Tensor) -> Tensor:
    """L_flow = SmoothL1(P_dense, P_GT)."""
    return F.smooth_l1_loss(pred, target)


def diffusion_noise_loss(pred_noise: Tensor, target_noise: Tensor) -> Tensor:
    """L_LDM stub."""
    return F.mse_loss(pred_noise, target_noise)


def total_loss(
    l_ldm: Tensor,
    l_shift: Tensor,
    l_flow: Tensor,
    *,
    lambda_shift: float,
    lambda_flow: float,
) -> Tensor:
    return l_ldm + lambda_shift * l_shift + lambda_flow * l_flow
