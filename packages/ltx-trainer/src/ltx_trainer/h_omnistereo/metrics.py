"""Disparity and normal evaluation metrics (Sec. IV-B)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def mean_absolute_error(pred: Tensor, gt: Tensor) -> float:
    return float(torch.mean(torch.abs(pred - gt)).item())


def root_mean_square_error(pred: Tensor, gt: Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - gt) ** 2)).item())


def bad_pixel_fraction(pred: Tensor, gt: Tensor, threshold: float) -> float:
    mask = torch.abs(pred - gt) >= threshold
    return float(100.0 * mask.float().mean().item())


def d1_outlier_fraction(pred: Tensor, gt: Tensor) -> float:
    """Outlier % with |e|≥3 and relative error ≥5% (paper D1)."""
    err = torch.abs(pred - gt)
    rel = err / gt.abs().clamp(min=1e-3)
    bad = (err >= 3.0) & (rel >= 0.05)
    return float(100.0 * bad.float().mean().item())


def angular_error_deg(pred: Tensor, gt: Tensor) -> dict[str, float]:
    p = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
    t = torch.nn.functional.normalize(gt, dim=1, eps=1e-6)
    cos = (p * t).sum(dim=1).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return {
        "mae": float(ang.mean().item()),
        "rmse": float(torch.sqrt((ang**2).mean()).item()),
    }


def delta_accuracy(pred: Tensor, gt: Tensor, thresh_deg: float) -> float:
    p = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
    t = torch.nn.functional.normalize(gt, dim=1, eps=1e-6)
    cos = (p * t).sum(dim=1).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return float(100.0 * (ang < thresh_deg).float().mean().item())
