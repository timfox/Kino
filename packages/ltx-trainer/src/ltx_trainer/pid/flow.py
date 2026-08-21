"""Rectified flow helpers for PiD pixel-space decoding (Eq. 7–10)."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_noise_like(x: Tensor) -> Tensor:
    return torch.randn_like(x)


def corrupt_latent(
    z: Tensor,
    sigma: float,
    *,
    noise: Tensor | None = None,
) -> Tensor:
    """Eq. (3): z̃_σ = (1-σ)z + σξ."""
    s = float(sigma)
    xi = noise if noise is not None else sample_noise_like(z)
    return (1.0 - s) * z + s * xi


def interpolate_state(x0: Tensor, x1: Tensor, t: float) -> Tensor:
    """x_t = t·x0 + (1-t)·ε  with x1=clean, ε=noise — paper uses x_t = t·x0 + (1-t)·ε."""
    return t * x0 + (1.0 - t) * x1


def velocity_target(x0: Tensor, epsilon: Tensor) -> Tensor:
    """Rectified-flow target v* ≈ x0 - ε (Eq. 8)."""
    return x0 - epsilon


def flow_matching_loss(v_pred: Tensor, v_star: Tensor) -> Tensor:
    """Eq. (9)/(10) L_FM."""
    return (v_pred - v_star).pow(2).mean()


def integrate_velocity(
    x_noise: Tensor,
    velocities: list[Tensor],
    sigmas: tuple[float, ...],
) -> Tensor:
    """Euler integration along DMD2 sigma schedule (few-step student)."""
    schedule = list(sigmas)
    if len(velocities) != len(schedule) - 1:
        raise ValueError(
            f"expected {len(schedule) - 1} velocities for {len(schedule)} sigmas, got {len(velocities)}"
        )
    x = x_noise
    for i, v in enumerate(velocities):
        dt = schedule[i] - schedule[i + 1]
        x = x + v * dt
    return x.clamp(0.0, 1.0)
