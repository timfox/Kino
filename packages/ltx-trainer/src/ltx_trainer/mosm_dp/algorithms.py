"""DP-MultiGreedy and DP-Bicriteria toy oracles (arXiv:2606.05596)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def laplace_noise(scale: float, rng: np.random.Generator) -> float:
    return float(rng.laplace(0.0, scale))


def coverage_value(covers: Sequence[set[int]], picked: Sequence[int]) -> int:
    union: set[int] = set()
    for i in picked:
        union |= covers[i]
    return len(union)


def marginal_gain(
    covers: Sequence[set[int]],
    picked: Sequence[int],
    candidate: int,
) -> int:
    before = coverage_value(covers, picked)
    after = coverage_value(covers, list(picked) + [candidate])
    return after - before


def dp_greedy_single(
    covers: Sequence[set[int]],
    k: int,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """(ε,δ)-DP greedy on one coverage objective (Gupta-style noisy marginal)."""
    rng = rng or np.random.default_rng(0)
    n = len(covers)
    picked: list[int] = []
    scale = sensitivity / max(epsilon, 1e-9)
    for _ in range(k):
        best_i = -1
        best_score = -np.inf
        for i in range(n):
            if i in picked:
                continue
            gain = marginal_gain(covers, picked, i)
            score = gain + laplace_noise(scale, rng)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        picked.append(best_i)
    return picked


def minimax_objective(
    objectives: Sequence[Sequence[set[int]]],
    picked: Sequence[int],
) -> int:
    return min(coverage_value(obj, picked) for obj in objectives)


def dp_multi_greedy(
    objectives: Sequence[Sequence[set[int]]],
    k: int,
    *,
    epsilon: float = 1.0,
    epsilon_split: tuple[float, float] = (0.5, 0.5),
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Two-phase DP-MultiGreedy when d ≤ k."""
    rng = rng or np.random.default_rng(0)
    d = len(objectives)
    if d > k:
        raise ValueError("DP-MultiGreedy requires d ≤ k")
    eps1, eps2 = epsilon * epsilon_split[0], epsilon * epsilon_split[1]
    per_budget = max(1, k // d)
    phase1: list[list[int]] = []
    union: list[int] = []
    for j in range(d):
        sj = dp_greedy_single(
            objectives[j],
            per_budget,
            epsilon=eps1 / d,
            rng=rng,
        )
        phase1.append(sj)
        for x in sj:
            if x not in union:
                union.append(x)
    picked = list(union)
    scale = 1.0 / max(eps2, 1e-9)
    while len(picked) < k:
        best_i = -1
        best_score = -np.inf
        base = minimax_objective(objectives, picked)
        for i in range(len(objectives[0])):
            if i in picked:
                continue
            trial = picked + [i]
            gain = minimax_objective(objectives, trial) - base
            score = gain + laplace_noise(scale, rng)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        picked.append(best_i)
    picked = picked[:k]
    return {
        "picked": picked,
        "phase1_sizes": [len(s) for s in phase1],
        "F": minimax_objective(objectives, picked),
        "d": d,
        "k": k,
    }


def dp_bicriteria_smoke(
    objectives: Sequence[Sequence[set[int]]],
    k: int,
    *,
    epsilon: float = 1.0,
    alpha: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Bucketed bicriteria: report (α,ε)-style split between utility and privacy budget."""
    rng = rng or np.random.default_rng(1)
    utility_eps = epsilon * alpha
    out = dp_multi_greedy(objectives, k, epsilon=utility_eps, rng=rng)
    out["bicriteria_alpha"] = alpha
    out["utility_epsilon"] = utility_eps
    out["privacy_epsilon"] = epsilon
    return out


def toy_three_objective_instance(seed: int = 0) -> list[list[set[int]]]:
    """Small synthetic MOSM instance (coverage per community)."""
    rng = np.random.default_rng(seed)
    n = 24
    m = 6
    objectives: list[list[set[int]]] = []
    for _ in range(3):
        covers: list[set[int]] = []
        for _i in range(n):
            size = int(rng.integers(1, 4))
            covers.append(set(rng.choice(m, size=size, replace=False).tolist()))
        objectives.append(covers)
    return objectives
