"""Flow Matching Distillation for 4D-GS (Sec. 3.3, Eq. 12–13)."""

from __future__ import annotations

import torch
from torch import Tensor


def corrupt_latent(z: Tensor, tau: Tensor, eps: Tensor | None = None) -> Tensor:
    """z_τ = (1 - τ) z + τ ε (Eq. 12)."""
    if eps is None:
        eps = torch.randn_like(z)
    while tau.dim() < z.dim():
        tau = tau.unsqueeze(-1)
    return (1.0 - tau) * z + tau * eps


def clean_estimate(z_tau: Tensor, tau: Tensor, v_pred: Tensor) -> Tensor:
    """f_Θ = z_τ - τ v_Θ (rectified-flow denoising estimate)."""
    while tau.dim() < z_tau.dim():
        tau = tau.unsqueeze(-1)
    return z_tau - tau * v_pred


def fmd_loss(
    z_clean: Tensor,
    z_hat: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    """L_FMD = ω(τ) ||f_Θ - z||^2 (Eq. 13)."""
    diff = (z_hat - z_clean) ** 2
    if weight is not None:
        while weight.dim() < diff.dim():
            weight = weight.unsqueeze(-1)
        diff = diff * weight
    return diff.mean()
