"""Calibration and uncertainty quality metrics (Sec. 4.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def spread_to_skill_ratio(samples: Tensor, target: Tensor, *, k: int | None = None) -> float:
    """SSR: ensemble spread / RMSE; ideal = 1 (calibrated)."""
    if samples.dim() == 1:
        samples = samples.unsqueeze(-1)
        target = target.unsqueeze(-1)
    k = k or samples.shape[0]
    mean_pred = samples.mean(dim=0)
    skill = torch.sqrt(((mean_pred - target) ** 2).mean()).item()
    if skill <= 1e-12:
        return 1.0
    spread = samples.std(dim=0, unbiased=False).mean().item()
    # finite-K correction factor from paper (approximate)
    correction = math.sqrt(1.0 + 1.0 / max(k, 1))
    return float(spread * correction / skill)


def uncertainty_spearman(errors: Tensor, variances: Tensor) -> float:
    """Spearman ρ between squared error and predicted variance."""
    if errors.numel() < 2:
        return 0.0
    err = errors.detach().flatten().cpu()
    var = variances.detach().flatten().cpu()
    err_rank = err.argsort().argsort().float()
    var_rank = var.argsort().argsort().float()
    err_c = err_rank - err_rank.mean()
    var_c = var_rank - var_rank.mean()
    denom = err_c.norm() * var_c.norm()
    if denom <= 1e-12:
        return 0.0
    return float((err_c * var_c).sum() / denom)
