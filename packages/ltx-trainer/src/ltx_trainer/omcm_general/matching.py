"""Pool-based online matching on the line (toy general-arrivals simulator)."""

from __future__ import annotations

from typing import Any

import numpy as np


def matching_cost(pos_a: float, pos_b: float) -> float:
    return abs(pos_a - pos_b)


def simulate_general_arrivals(
    positions: list[float],
    *,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Both sides arrive online; greedy match to nearest pool partner within threshold."""
    pool: list[tuple[int, float]] = []
    total_cost = 0.0
    matched_pairs = 0
    for idx, x in enumerate(positions):
        if not pool:
            pool.append((idx, x))
            continue
        best_j = -1
        best_d = np.inf
        for j, (_, y) in enumerate(pool):
            d = matching_cost(x, y)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= threshold:
            pool.pop(best_j)
            total_cost += best_d
            matched_pairs += 1
        else:
            pool.append((idx, x))
    return {
        "total_cost": total_cost,
        "matched_pairs": matched_pairs,
        "unmatched_pool": len(pool),
        "n": len(positions),
    }


def competitive_ratio_bound(n: int, *, model: str = "unknown_iid") -> str:
    if model == "unknown_iid":
        c = max(1, int(np.ceil(np.log2(max(n, 2)) ** 2)))
        return f"O(log² n) ≈ {c} (toy scale)"
    if model == "random_order":
        return "unbounded (Theorem separation)"
    return "unknown"


def toy_arrival_stream(seed: int = 0, n: int = 20) -> list[float]:
    rng = np.random.default_rng(seed)
    return sorted(rng.uniform(0.0, 1.0, size=n).tolist())
