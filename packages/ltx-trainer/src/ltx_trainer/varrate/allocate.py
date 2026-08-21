"""VarRate salience + water-filling rank allocation (no eviction)."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ltx_trainer.varrate.config import VarRateConfig


def snapkv_salience(n_tokens: int, *, window: int = 64, seed: int = 0) -> list[float]:
    """Deterministic SnapKV-style observation salience stub.

    Recent tokens get higher mass; older tokens decay. No model weights.
    """
    n = max(1, int(n_tokens))
    w = max(1, int(window))
    # Mix seed lightly so demos are reproducible but not identical across seeds.
    bias = 0.01 * (seed % 7)
    scores: list[float] = []
    for t in range(n):
        # Distance from end (0 = most recent).
        dist = (n - 1 - t) / max(1, n - 1)
        recent = 1.0 / (1.0 + dist * (n / w))
        scores.append(float(recent + bias * math.sin(t + 1)))
    return scores


def water_fill_ranks(
    salience: Sequence[float],
    *,
    budget: int,
    cfg: VarRateConfig | None = None,
) -> dict[str, Any]:
    """Allocate per-token ranks ``r_t`` with floor ``rmin`` and sum budget ``B``.

    Paper form: ``r_t = clip(rmin + λ ŝ_t, rmin, R)`` with ``Σ r_t = B``.
    No tokens are evicted — ranks may drop to ``rmin`` but stay present.
    """
    c = cfg or VarRateConfig()
    n = len(salience)
    if n == 0:
        return {"ranks": [], "budget": 0, "lambda": 0.0, "sum_ranks": 0, "evictions": 0}

    rmin = int(c.rmin)
    rmax = int(c.r_max)
    b = max(n * rmin, int(budget))

    s = [max(0.0, float(x)) for x in salience]
    total = sum(s) or 1.0
    s_hat = [x / total for x in s]

    # Solve λ so sum clip(rmin + λ ŝ, rmin, R) ≈ B (binary search).
    lo, hi = 0.0, float(n * rmax)
    best = [rmin] * n
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        ranks = [int(round(min(rmax, max(rmin, rmin + mid * sh)))) for sh in s_hat]
        ssum = sum(ranks)
        best = ranks
        if ssum > b:
            hi = mid
        else:
            lo = mid
        if abs(ssum - b) <= max(1, n // 50):
            break

    # Final nudge: distribute residual without dropping below rmin / above rmax.
    ssum = sum(best)
    residual = b - ssum
    idx = sorted(range(n), key=lambda i: s_hat[i], reverse=True)
    step = 1 if residual > 0 else -1
    guard = 0
    while residual != 0 and guard < n * rmax:
        for i in idx:
            if residual == 0:
                break
            nxt = best[i] + step
            if rmin <= nxt <= rmax:
                best[i] = nxt
                residual -= step
            guard += 1

    return {
        "ranks": best,
        "budget": b,
        "lambda": lo,
        "sum_ranks": sum(best),
        "rmin": rmin,
        "r_max": rmax,
        "evictions": 0,
        "mean_rank": sum(best) / n,
        "min_rank": min(best),
        "max_rank": max(best),
    }


def allocate_for_sequence(
    n_tokens: int,
    *,
    budget: int | None = None,
    cfg: VarRateConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    c = cfg or VarRateConfig()
    n = max(1, int(n_tokens))
    sal = snapkv_salience(n, window=c.window_obs, seed=seed)
    # Default budget: kappa * n * R (fraction of full-rank capacity).
    b = int(budget) if budget is not None else max(n * c.rmin, int(round(c.kappa * n * c.r_max)))
    alloc = water_fill_ranks(sal, budget=b, cfg=c)
    return {
        "n_tokens": n,
        "salience_head": sal[: min(5, n)],
        "allocation": alloc,
        "kappa": c.kappa,
        "no_eviction": alloc["evictions"] == 0 and alloc["min_rank"] >= c.rmin,
    }
