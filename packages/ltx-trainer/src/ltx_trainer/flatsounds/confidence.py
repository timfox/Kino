"""Physical correctness confidence voting (Sec. 5.2, App. E.13)."""

from __future__ import annotations

from typing import Literal

import numpy as np

Direction = Literal["increase", "decrease", "no_change", "ascending", "descending"]


def effect_threshold(values: np.ndarray, *, frac: float = 0.02, mad_mult: float = 0.25) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) + 1e-9)
    return max(frac * abs(med), mad_mult * mad)


def pair_direction_vote(
    value_a: float,
    value_b: float,
    expected: Direction,
    *,
    tau: float | None = None,
) -> bool:
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return False
    delta = value_b - value_a
    if tau is None:
        tau = effect_threshold(np.array([value_a, value_b]))
    if expected == "increase":
        return delta > tau
    if expected == "decrease":
        return delta < -tau
    if expected == "no_change":
        return abs(delta) <= tau
    return False


def monotonic_vote(hit_values: list[float], expected: Direction) -> bool:
    v = [x for x in hit_values if np.isfinite(x)]
    n = len(v)
    if n < 2:
        return False
    if n == 2:
        diff = v[1] - v[0]
        if expected == "ascending":
            return diff > 0
        if expected == "descending":
            return diff < 0
        return False
    ranks = np.argsort(np.argsort(v))
    expected_ranks = np.arange(n) if expected == "ascending" else np.arange(n)[::-1]
    rho = np.corrcoef(ranks, expected_ranks)[0, 1]
    thresh = 0.40 if n <= 4 else (0.30 if n <= 7 else 0.25)
    return bool(abs(rho) >= thresh and (
        (expected == "ascending" and rho > 0) or (expected == "descending" and rho < 0)
    ))


def weighted_confidence(
    votes: list[bool],
    weights: list[float],
) -> float:
    if not votes:
        return 0.0
    num = sum(w for v, w in zip(votes, weights, strict=True) if v)
    den = sum(weights) if weights else len(votes)
    return float(num / den) if den > 0 else 0.0


def quality_weight(
    hit_coverage_a: float,
    hit_coverage_b: float,
    clap_a: float,
    clap_b: float,
) -> float:
    """Soft gate: 0.5 temporal + 0.5 semantic (App. E.13)."""
    w_t = min(hit_coverage_a, hit_coverage_b)
    w_s = min(clap_a, clap_b)
    return 0.5 * w_t + 0.5 * w_s
