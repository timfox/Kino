"""Noise schedules: variance alignment, logit-normal t, PolyShift."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def scale_waveform(x: Tensor, k: float) -> Tensor:
    """Signal-noise variance alignment: x' = k * x1."""
    return x * k


def sample_timestep_logit_normal(
    batch: int,
    *,
    mu: float = -0.8,
    sigma: float = 0.8,
    device: torch.device | None = None,
) -> Tensor:
    """t ~ LogitNormal via u ~ N(mu, sigma^2), t = sigmoid(u)."""
    u = torch.randn(batch, device=device) * sigma + mu
    return torch.sigmoid(u)


def polyshift_schedule(nfe: int, *, p: float = 2.0, s: float = 3.0) -> Tensor:
    """Eq. (10): t = τ^p / (τ^p + s(1-τ^p)) for uniform τ in [0,1]."""
    tau = torch.linspace(0.0, 1.0, nfe + 1)
    tp = tau.pow(p)
    return tp / (tp + s * (1.0 - tp))


def log_snr(t: Tensor, sigma_x1: float, sigma_x0: float = 1.0) -> Tensor:
    """Eq. (8) log-SNR in dB."""
    ratio = (t * sigma_x1) / ((1.0 - t).clamp(min=1e-6) * sigma_x0)
    return 20.0 * torch.log10(ratio.clamp(min=1e-8))
