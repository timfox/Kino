"""Ornstein–Uhlenbeck attention-logit moments (paper Eq. 3–4, Appendix A.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def ou_mean_variance(
    a0: Tensor,
    kappa: Tensor,
    phi: Tensor,
    psi: Tensor,
    t: Tensor,
    *,
    kappa_floor: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Mean and variance of OU logit state at (frozen) time t.

    ``E[a_t] = φ + (a0 - φ) exp(-κ t)``, ``Var[a_t] = (ψ²/(2κ))(1 - exp(-2κ t))``.

    Shapes broadcast like PyTorch elementwise ops.
    """
    k = torch.clamp(kappa, min=kappa_floor)
    kt = k * t
    exp_mkt = torch.exp(-kt)
    mean = phi + (a0 - phi) * exp_mkt
    exp_m2kt = torch.exp(-2.0 * kt)
    var = (psi * psi) / (2.0 * k) * (1.0 - exp_m2kt)
    return mean, var.clamp_min(1e-12)
