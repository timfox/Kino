"""Flow matching for dual-stream AV editing (Eq. 1–3)."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_noise_like(x: Tensor) -> Tensor:
    return torch.randn_like(x)


def build_noisy_latents(
    z1_v: Tensor,
    z1_a: Tensor,
    t: float,
    *,
    eps_v: Tensor | None = None,
    eps_a: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eq. (1): z_t = (1-t)ε + t·z_1 for video and audio."""
    eps_v = eps_v if eps_v is not None else sample_noise_like(z1_v)
    eps_a = eps_a if eps_a is not None else sample_noise_like(z1_a)
    zt_v = (1.0 - t) * eps_v + t * z1_v
    zt_a = (1.0 - t) * eps_a + t * z1_a
    return zt_v, zt_a, eps_v, eps_a


def velocity_target(z1: Tensor, epsilon: Tensor) -> Tensor:
    """u_t = z_1 - ε."""
    return z1 - epsilon


def dual_flow_loss(
    u_hat_v: Tensor,
    u_hat_a: Tensor,
    u_v: Tensor,
    u_a: Tensor,
    *,
    lambda_v: float = 0.85,
    lambda_a: float = 0.15,
) -> Tensor:
    """Eq. (3) L_FM with modality weights."""
    lv = (u_hat_v - u_v).pow(2).mean()
    la = (u_hat_a - u_a).pow(2).mean()
    return lambda_v * lv + lambda_a * la
