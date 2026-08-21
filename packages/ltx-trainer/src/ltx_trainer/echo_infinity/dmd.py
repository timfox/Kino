"""Distribution Matching Distillation stub (Echo-Infinity §3.1, Eq. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def score_from_noise(
    x_t: Tensor,
    mu: Tensor,
    t: Tensor,
    *,
    alpha_t: float = 0.99,
    sigma_t: float = 0.01,
) -> Tensor:
    """s(x_t, t) = -(x_t - α_t μ) / σ_t²  (Eq. 2, simplified scalar schedule)."""
    alpha = torch.as_tensor(alpha_t, device=x_t.device, dtype=x_t.dtype)
    sigma = torch.as_tensor(sigma_t, device=x_t.device, dtype=x_t.dtype)
    return -(x_t - alpha * mu) / (sigma * sigma + 1e-8)


def dmd_generator_gradient(
    s_real: Tensor,
    s_fake: Tensor,
) -> Tensor:
    """∇_θ D_KL ≈ ∫ (s_real - s_fake) dG/dϵ  — return per-sample mismatch for smoke."""
    return s_real - s_fake


def dmd_step_loss(
    x_t_fake: Tensor,
    mu_real: Tensor,
    mu_fake: Tensor,
    t: Tensor,
) -> Tensor:
    """Smoke loss: L2 between real/fake score fields."""
    sr = score_from_noise(x_t_fake, mu_real, t)
    sf = score_from_noise(x_t_fake, mu_fake, t)
    grad = dmd_generator_gradient(sr, sf)
    return (grad * grad).mean()
