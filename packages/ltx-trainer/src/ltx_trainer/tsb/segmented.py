"""Batch / online segmented beamforming — §V, Eqs. (24)–(26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.tsb.mvdr import mvdr_weights, sample_covariance, segment_output_power


def bellman_segment_cost(
    segment_powers: list[list[float]],
    penalty_c: float,
) -> tuple[list[float], list[int]]:
    """Penalized DP — Eq. (25); segment_powers[i][j] = E(i+1, j+1) 0-indexed."""
    t = len(segment_powers)
    e = [float("inf")] * (t + 1)
    back: list[int] = [-1] * (t + 1)
    e[0] = 0.0
    for j in range(1, t + 1):
        for i in range(1, j + 1):
            cost = segment_powers[i - 1][j - 1] + penalty_c + e[i - 1]
            if cost < e[j]:
                e[j] = cost
                back[j] = i - 1
    return e, back


def traceback_partitions(back: list[int], t: int) -> list[tuple[int, int]]:
    """Recover segment boundaries from DP backpointers."""
    parts: list[tuple[int, int]] = []
    j = t
    while j > 0:
        i = back[j] + 1
        parts.append((i, j))
        j = back[j]
    parts.reverse()
    return parts


def batch_segmented_beamformer(
    snapshots: np.ndarray,
    steering: np.ndarray,
    penalty_c: float,
    delta: float = 1e-3,
) -> dict[str, Any]:
    """Toy BSB — Algorithm 5 structure."""
    _, t = snapshots.shape
    powers = [[0.0] * t for _ in range(t)]
    for i in range(t):
        for j in range(i, t):
            seg = snapshots[:, i : j + 1]
            powers[i][j] = segment_output_power(seg, steering, delta)
    costs, back = bellman_segment_cost(powers, penalty_c)
    parts = traceback_partitions(back, t)
    z = np.zeros(t)
    for i, j in parts:
        seg = snapshots[:, i:j]
        w = mvdr_weights(sample_covariance(seg, delta), steering, delta)
        z[i:j] = w @ seg
    return {
        "output": z,
        "partitions": parts,
        "penalized_cost": costs[t],
        "num_segments": len(parts),
    }


def online_segment_step(
    segment_costs: list[float],
    frozen_cost: float,
    penalty_c: float,
    cur: int,
    best: int,
    tau: int,
) -> tuple[int, bool]:
    """OSB change-point decision — Algorithm 6."""
    emin = float("inf")
    best_i = cur
    for i, jc in enumerate(segment_costs):
        total = frozen_cost + penalty_c + jc
        if total < emin:
            emin = total
            best_i = cur + i
    changed = (best_i - cur) > tau and best_i != cur
    return best_i, changed
