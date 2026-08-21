"""Diffusion DPO for physics preference pairs (Sec. 3.2)."""

from __future__ import annotations

import math

import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def diffusion_dpo_loss(
    mse_policy_winner: float,
    mse_policy_loser: float,
    mse_ref_winner: float,
    mse_ref_loser: float,
    *,
    beta: float = 100.0,
) -> float:
    """L_DPO = -log σ(β Δ) with Δ = (MSE_π,l - MSE_π,w) - (MSE_ref,l - MSE_ref,w)."""
    delta = (mse_policy_loser - mse_policy_winner) - (mse_ref_loser - mse_ref_winner)
    return -math.log(sigmoid(beta * delta) + 1e-12)


def preference_logit(
    policy_winner_loss: float,
    policy_loser_loss: float,
    ref_winner_loss: float = 0.0,
    ref_loser_loss: float = 0.0,
) -> float:
    """DPO preference logit: positive when the winner is easier than the loser under policy vs reference."""
    return (policy_loser_loss - policy_winner_loss) - (ref_loser_loss - ref_winner_loss)


def aggregate_human_score(
    sa: float,
    ptv: float,
    persistence: float,
) -> float:
    """Per-video quality s(v) on 3–15 summed-axis scale (Sec. 3.2)."""
    return sa + ptv + persistence


def preference_margin_ok(winner_score: float, loser_score: float, *, min_margin: float = 1.0) -> bool:
    return winner_score - loser_score >= min_margin


def sample_timestep_high_noise(
    rng: np.random.Generator,
    *,
    t_min: int = 901,
    t_max: int = 999,
) -> int:
    return int(rng.integers(t_min, t_max + 1))
