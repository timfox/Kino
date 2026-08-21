"""Parametric camera response (Eilertsen et al. 2017) for SDR simulation and inversion."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_training_crf_params(
    batch: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    n_mean: float = 0.9,
    n_std: float = 0.1,
    sigma_mean: float = 0.6,
    sigma_std: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Sample ``(n, sigma)`` for ``f(H) = (1+sigma) H^n / (H^n + sigma)`` (paper Supp. S4)."""
    n = torch.randn(batch, device=device, dtype=dtype) * n_std + n_mean
    sigma = torch.randn(batch, device=device, dtype=dtype) * sigma_std + sigma_mean
    return n.clamp(min=0.05), sigma.clamp(min=1e-3)


def apply_crf_forward(linear: Tensor, n: Tensor, sigma: Tensor) -> Tensor:
    """Apply parametric CRF to scene-linear ``linear`` in ``[0, inf)`` → ``[0, 1]``."""
    x = linear.clamp(min=0.0)
    if n.ndim == 0:
        n = n.view(1)
    if sigma.ndim == 0:
        sigma = sigma.view(1)
    while n.ndim < x.ndim:
        n = n.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    xn = x.pow(n)
    return ((1.0 + sigma) * xn / (xn + sigma)).clamp(0.0, 1.0)


def estimate_inverse_crf_linear(
    sdr_gamma: Tensor,
    *,
    gamma: float = 2.2,
    n: float = 0.9,
    sigma: float = 0.6,
    iters: int = 8,
) -> Tensor:
    """Invert gamma + parametric CRF via fixed-point iteration on ``H``.

    Args:
        sdr_gamma: Display-encoded SDR ``[..., C, ...]`` in ``[0, 1]``.
    """
    y = sdr_gamma.clamp(0.0, 1.0).pow(gamma)
    h = y.clone()
    n_t = torch.tensor(n, device=y.device, dtype=y.dtype)
    sig_t = torch.tensor(sigma, device=y.device, dtype=y.dtype)
    for _ in range(iters):
        hn = h.clamp(min=1e-8).pow(n_t)
        h = (y * (hn + sig_t) / ((1.0 + sig_t) * n_t * hn.clamp(min=1e-8))).clamp(min=0.0, max=1.0)
    return h
