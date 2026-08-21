"""Log-Gamma color mapping (Sec. 3.2, Eq. 3)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.diffhdr.config import LOG_GAMMA_GAMMA, LOG_GAMMA_M


def log_gamma_map(
    x: Tensor,
    *,
    m: float = LOG_GAMMA_M,
    gamma: float = LOG_GAMMA_GAMMA,
) -> Tensor:
    """T(x) = (log(1+γx) / log(1+γM))^(1/γ)."""
    x = x.clamp(min=0.0)
    num = torch.log1p(gamma * x)
    den = math.log1p(gamma * m)
    return (num / den) ** (1.0 / gamma)


def inverse_log_gamma_map(
    y: Tensor,
    *,
    m: float = LOG_GAMMA_M,
    gamma: float = LOG_GAMMA_GAMMA,
) -> Tensor:
    """Invert Log-Gamma mapping to recover linear HDR radiance."""
    y = y.clamp(0.0, 1.0)
    log_denom = math.log1p(gamma * m)
    inner = y**gamma
    return (torch.exp(inner * log_denom) - 1.0) / gamma


def log_map_only(x: Tensor, *, m: float = LOG_GAMMA_M) -> Tensor:
    """Log mapping without gamma (ablation baseline)."""
    x = x.clamp(min=0.0)
    den = math.log1p(m)
    return torch.log1p(x) / den
