"""COCONUT-style latent reasoning + alignment (§2.1)."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.latent_prm_guidance.constants import PERTURBATION_STD


def alignment_transform_card() -> dict[str, Any]:
    return {
        "reference": "Zou et al. 2025 training-free alignment",
        "formula": "Wa = (Wout^T Wout)^(-1) Wout^T Win",
        "latent_input": "z_{t+1} = h_t Wa",
        "primary_frozen": True,
        "note": "h_t not decoded during latent phase; fed back as continuous input",
    }


def perturb_branch(
    h_t: list[float],
    *,
    sigma: float = PERTURBATION_STD,
    rng: random.Random | None = None,
) -> list[float]:
    """Eq. 1: h̃_t = h_t + ε, ε ~ N(0, σ² I)."""
    rng = rng or random.Random(0)
    return [x + rng.gauss(0.0, sigma) for x in h_t]


def sample_branches(
    h_t: list[float],
    n_branches: int,
    *,
    include_unperturbed: bool = True,
    sigma: float = PERTURBATION_STD,
    seed: int = 0,
) -> list[list[float]]:
    """Sample candidate branch states at latent step t."""
    rng = random.Random(seed + len(h_t))
    branches: list[list[float]] = []
    if include_unperturbed:
        branches.append(list(h_t))
    while len(branches) < n_branches:
        branches.append(perturb_branch(h_t, sigma=sigma, rng=rng))
    return branches[:n_branches]


def greedy_branch_select(
    prefix_scores: list[tuple[int, float]],
) -> int:
    """b* = argmax_b Vθ(τ^b_≤t) over scored candidates."""
    if not prefix_scores:
        raise ValueError("prefix_scores must be non-empty")
    return max(prefix_scores, key=lambda x: x[1])[0]
