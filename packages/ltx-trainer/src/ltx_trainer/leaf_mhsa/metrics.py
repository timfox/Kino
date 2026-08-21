"""R², NRMSE, MAE (Eq. 3–5)."""

from __future__ import annotations

import torch
from torch import Tensor


def mse(y: Tensor, y_hat: Tensor) -> Tensor:
    return torch.mean((y - y_hat) ** 2)


def r2_score(y: Tensor, y_hat: Tensor) -> Tensor:
    ss_res = torch.sum((y - y_hat) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    if ss_tot <= 0:
        return torch.tensor(0.0, device=y.device)
    return 1.0 - ss_res / ss_tot


def nrmse_percent(y: Tensor, y_hat: Tensor) -> Tensor:
    err = torch.sqrt(mse(y, y_hat))
    span = torch.max(y) - torch.min(y)
    if span <= 0:
        return torch.tensor(0.0, device=y.device)
    return 100.0 * err / span


def mae_per_band(y: Tensor, y_hat: Tensor) -> Tensor:
    """Mean absolute error per spectral band; y, y_hat shape (B, L)."""
    return torch.mean(torch.abs(y - y_hat), dim=0)
