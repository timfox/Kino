"""Scaled-linear β schedule and LCM timestep helpers."""

from __future__ import annotations

import numpy as np


def scaled_linear_betas(num_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012) -> np.ndarray:
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def diffusion_schedule(num_timesteps: int = 1000) -> dict[str, np.ndarray]:
    betas = scaled_linear_betas(num_timesteps)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


def add_noise(
    x0: np.ndarray,
    noise: np.ndarray,
    timestep: int,
    *,
    num_timesteps: int = 1000,
) -> np.ndarray:
    sched = diffusion_schedule(num_timesteps)
    ab = float(sched["alpha_bar"][int(np.clip(timestep, 0, num_timesteps - 1))])
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise


def lcm_inference_timesteps(num_steps: int, *, num_train_steps: int = 1000) -> list[int]:
    """Uniform stride indices from T down to 0 (inclusive endpoints)."""
    if num_steps <= 1:
        return [num_train_steps - 1, 0]
    return [int(round(i)) for i in np.linspace(num_train_steps - 1, 0, num_steps)]
