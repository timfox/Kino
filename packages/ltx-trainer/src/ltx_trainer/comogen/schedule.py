"""Reverse cosine latent injection schedule (Sec. 4.3, Eq. 3–4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_step_weight(step: int, total_steps: int) -> float:
    """w_s = 0.5 * (1 + cos(π * t_s)), t_s = s / (τ - 1)."""
    if total_steps <= 1:
        return 1.0
    ts = step / (total_steps - 1)
    return 0.5 * (1.0 + math.cos(math.pi * ts))


def cosine_schedule(total_steps: int) -> list[float]:
    return [cosine_step_weight(s, total_steps) for s in range(total_steps)]


def inject_latent_residual(
    z_t: Tensor,
    delta_z: Tensor,
    step: int,
    total_steps: int,
) -> Tensor:
    """Z̃(s)_t = Z(s)_t + w_s · ΔZ (Eq. 4)."""
    w = cosine_step_weight(step, total_steps)
    return z_t + w * delta_z


def inject_schedule_tensor(z_t: Tensor, delta_z: Tensor, weights: Tensor) -> Tensor:
    """Apply per-step weights [τ] to batched latents [τ, ...]."""
    w = weights.to(dtype=z_t.dtype, device=z_t.device)
    while w.dim() < z_t.dim():
        w = w.unsqueeze(-1)
    return z_t + w * delta_z.unsqueeze(0) if delta_z.dim() == z_t.dim() - 1 else z_t + w * delta_z
