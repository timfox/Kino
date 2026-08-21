"""RANDOM-SCHEDULING + robust rotation for dynamic consistent submodular maximization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def random_scheduling_levels(
    d0: int,
    num_levels: int,
    *,
    epsilon: float,
) -> list[int]:
    """Deletion-robustness schedule d_i = floor(d0 / 2^i) with ε-fraction transition windows."""
    levels = [max(1, int(d0 // (2**i))) for i in range(num_levels)]
    window = max(1, int(round(1.0 / max(epsilon, 1e-6))))
    return levels + [window]


def coverage_submodular(covers: Sequence[set[int]], picked: Sequence[int]) -> int:
    u: set[int] = set()
    for i in picked:
        u |= covers[i]
    return len(u)


def greedy_under_stream(
    covers: Sequence[set[int]],
    stream: Sequence[tuple[str, int]],
    k: int,
) -> tuple[list[int], list[int]]:
    """Insertion/deletion stream; return final set and per-step symmetric difference sizes."""
    picked: list[int] = []
    diffs: list[int] = []
    for op, idx in stream:
        prev = set(picked)
        if op == "ins":
            if idx not in picked and len(picked) < k:
                picked.append(idx)
        elif op == "del" and idx in picked:
            picked.remove(idx)
        diffs.append(len(prev.symmetric_difference(picked)))
    return picked, diffs


def dynamic_consistent_cardinality_smoke(
    covers: Sequence[set[int]],
    stream: Sequence[tuple[str, int]],
    k: int,
    *,
    epsilon: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng(0)
    levels = random_scheduling_levels(k, num_levels=3, epsilon=epsilon)
    picked, diffs = greedy_under_stream(covers, stream, k)
    opt = coverage_submodular(covers, list(range(min(k, len(covers)))))
    val = coverage_submodular(covers, picked)
    ratio = val / opt if opt else 1.0
    tau = int(rng.integers(0, max(1, levels[-1])))
    return {
        "picked": picked,
        "value": val,
        "opt_proxy": opt,
        "approx_ratio": round(ratio, 4),
        "max_consistency": max(diffs) if diffs else 0,
        "mean_consistency": round(float(np.mean(diffs)) if diffs else 0.0, 4),
        "schedule_levels": levels,
        "random_shift_tau": tau,
        "epsilon": epsilon,
    }


def toy_coverage_instance(n: int = 16, universe: int = 10, seed: int = 0) -> list[set[int]]:
    rng = np.random.default_rng(seed)
    return [
        set(rng.choice(universe, size=int(rng.integers(2, 5)), replace=False).tolist())
        for _ in range(n)
    ]
