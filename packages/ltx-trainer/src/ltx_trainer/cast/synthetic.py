"""Synthetic distribution-valued time series and aliasing experiment."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cast.simplex import normalize_simplex
from ltx_trainer.cast.transport import apply_local_transport


def _random_simplex(batch: int, dim: int, gen: torch.Generator) -> Tensor:
    x = torch.rand(batch, dim, generator=gen)
    return normalize_simplex(x)


def queue_occupancy_sequence(
    batch: int = 4,
    length: int = 32,
    dim: int = 32,
    *,
    seed: int | None = None,
) -> Tensor:
    """Synthetic ordered queue occupancy distributions drifting locally."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    seq = []
    state = _random_simplex(batch, dim, gen)
    seq.append(state)
    for _ in range(length - 1):
        logits = torch.zeros(batch, dim, 3)
        logits[:, :, 1] = 2.0  # prefer stay
        logits[:, :, 0] = 0.5
        logits[:, :, 2] = 0.5
        state = apply_local_transport(state, logits + torch.randn(batch, dim, 3, generator=gen) * 0.1)
        seq.append(state)
    return torch.stack(seq, dim=1)


def compositional_sequence(
    batch: int = 4,
    length: int = 24,
    dim: int = 16,
    *,
    seed: int | None = None,
) -> Tensor:
    """Smooth compositional drift on unordered support."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    seq = []
    state = _random_simplex(batch, dim, gen)
    seq.append(state)
    for _ in range(length - 1):
        noise = torch.randn(batch, dim, generator=gen) * 0.05
        state = normalize_simplex(0.85 * state + 0.15 * _random_simplex(batch, dim, gen) + noise.abs())
        seq.append(state)
    return torch.stack(seq, dim=1)


def aliasing_setup(
    dim: int = 48,
    rho: float = 0.15,
) -> dict[str, Tensor]:
    """
    Controlled aliasing experiment (Sec. 6.2, Table 5).

    Same p*, opposite left/right local transports under two regimes.
    """
    center = dim // 3
    p_star = torch.zeros(dim)
    p_star[center - 2 : center + 3] = 1.0
    p_star = normalize_simplex(p_star)

    def _shift_kernel(direction: int) -> Tensor:
        logits = torch.zeros(1, dim, 3)
        if direction > 0:
            logits[:, :, 2] = 3.0
        else:
            logits[:, :, 0] = 3.0
        logits[:, :, 1] = 1.0
        return logits

    t_right = apply_local_transport(p_star.unsqueeze(0), _shift_kernel(1)).squeeze(0)
    t_left = apply_local_transport(p_star.unsqueeze(0), _shift_kernel(-1)).squeeze(0)
    u_up = normalize_simplex((1 - rho) * p_star + rho * t_right)
    u_down = normalize_simplex((1 - rho) * p_star + rho * t_left)
    mixture = normalize_simplex(0.5 * u_up + 0.5 * u_down)
    return {
        "p_star": p_star,
        "u_up": u_up,
        "u_down": u_down,
        "mixture": mixture,
    }


def dataset_summary() -> dict:
    return {
        "name": "CAST 11-section benchmark suite",
        "sections": 11,
        "domains": [
            "ecology",
            "energy",
            "diet",
            "mortality",
            "employment",
            "air quality",
            "severe weather",
            "mobility",
            "queueing",
        ],
        "metrics": ["offline KL", "rollout JSD", "L1", "W1 (ordered)"],
        "primary_one_step": "offline KL",
        "primary_rollout": "rollout JSD",
    }
