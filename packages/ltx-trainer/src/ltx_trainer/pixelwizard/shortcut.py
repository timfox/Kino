"""Noise-Span Aligned Shortcut Training (Eq. 3–8, Sec. 3.4)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def shortcut_step(x_t: Tensor, velocity: Tensor, delta_t: Tensor) -> Tensor:
    """Eq. (3): x_{t+Δt} = x_t + Δt · s_θ(x_t, t, Δt)."""
    dt = delta_t.view(-1, *([1] * (x_t.dim() - 1)))
    return x_t + dt * velocity


def consistency_target(
    v_t: Tensor,
    v_t_dt: Tensor,
    x_t: Tensor,
    t: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """Eq. (4) stop-gradient target: average of two Δt steps."""
    x_next = shortcut_step(x_t, v_t, delta_t)
    return 0.5 * (v_t + v_t_dt)


def candidate_step_sizes(timestep: int, k: int = 6) -> list[int]:
    """Eq. (5): ΔT_k(T) = floor(T / 2^k) for k = 0..K-1."""
    return [max(1, timestep // (2**i)) for i in range(k)]


def sample_step_index(k: int = 6, beta: float = 0.7, *, generator: torch.Generator | None = None) -> int:
    """Eq. (6): p(k) ∝ exp(-βk), favoring large steps (small k)."""
    weights = torch.tensor([math.exp(-beta * j) for j in range(k)], dtype=torch.float32)
    probs = weights / weights.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


def shifted_sigma(timestep: int, n: int = 1000) -> float:
    """Proxy shifted noise schedule σ(t) for high-resolution flow matching."""
    t_norm = timestep / max(n - 1, 1)
    return float(0.02 + 0.98 * (t_norm**1.5))


def noise_span(timestep: int, delta_t: int, n: int = 1000) -> float:
    """Eq. (7): |σ(t+Δt) − σ(t)|."""
    return abs(shifted_sigma(min(timestep + delta_t, n - 1), n) - shifted_sigma(timestep, n))


def calibration_weight(timestep: int, delta_t: int, power: float = 0.5, n: int = 1000) -> float:
    """Eq. (8): λ(T, ΔT) = (Δσ)^{p}."""
    return noise_span(timestep, delta_t, n) ** power


def select_shortcut_step(
    timestep: int,
    *,
    k: int = 6,
    beta: float = 0.7,
    generator: torch.Generator | None = None,
) -> int:
    """Sample ΔT from D(T) with exponential index bias."""
    candidates = candidate_step_sizes(timestep, k)
    idx = sample_step_index(k, beta, generator=generator)
    return candidates[idx]


def shortcut_consistency_loss(
    pred_2dt: Tensor,
    target_2dt: Tensor,
    timestep: int,
    delta_t: int,
    power: float = 0.5,
) -> Tensor:
    """L_ANC = λ(T, ΔT) · L_sc (Eq. 8)."""
    lam = calibration_weight(timestep, delta_t, power)
    return lam * torch.nn.functional.mse_loss(pred_2dt, target_2dt)
