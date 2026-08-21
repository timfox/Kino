"""Photometric losses — intra face (Eq. 10) and inter ERP (Eq. 12)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def dssim_loss(pred: Tensor, target: Tensor, window: int = 3) -> Tensor:
    """Lightweight SSIM surrogate for CPU stub."""
    mu_p = F.avg_pool2d(pred, window, stride=1, padding=window // 2)
    mu_t = F.avg_pool2d(target, window, stride=1, padding=window // 2)
    sigma_p = F.avg_pool2d(pred * pred, window, stride=1, padding=window // 2) - mu_p * mu_p
    sigma_t = F.avg_pool2d(target * target, window, stride=1, padding=window // 2) - mu_t * mu_t
    sigma_pt = F.avg_pool2d(pred * target, window, stride=1, padding=window // 2) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (sigma_p + sigma_t + c2).clamp_min(1e-8)
    )
    return 1.0 - ssim.mean()


def photometric_loss(
    pred: Tensor,
    target: Tensor,
    *,
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
) -> Tensor:
    return lambda_l1 * l1_loss(pred, target) + lambda_dssim * dssim_loss(pred, target)


def stitch_erp(
    ec: Tensor,
    et: Tensor,
    *,
    shift_cols: int = 0,
) -> Tensor:
    """
    Eq. 11 stub: Er = (Rotate(Et, −45°) + Ec) / 2.
    shift_cols approximates horizontal roll alignment.
    """
    if shift_cols:
        et = torch.roll(et, shifts=-shift_cols, dims=-1)
    return (et + ec) / 2.0
