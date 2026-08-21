"""Normative ethics simplex — eq. (1): α + β + γ = 1 (arXiv:2605.28707)."""

from __future__ import annotations

import math
from typing import Any

NormativeScores = tuple[float, float, float]  # (α, β, γ)


def project_simplex(alpha: float, beta: float, gamma: float) -> NormativeScores:
    """Project nonnegative scores onto the unit simplex."""
    v = [max(0.0, alpha), max(0.0, beta), max(0.0, gamma)]
    s = sum(v)
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (v[0] / s, v[1] / s, v[2] / s)


def simplex_constraint_ok(scores: NormativeScores, tol: float = 1e-5) -> bool:
    a, b, g = scores
    return abs(a + b + g - 1.0) <= tol and min(a, b, g) >= -tol


def normative_entropy(scores: NormativeScores) -> float:
    """Shannon entropy over (α, β, γ) — high ⇒ pluralistic / ambiguous."""
    ent = 0.0
    for p in scores:
        if p > 1e-12:
            ent -= p * math.log(p)
    return ent


def score_margin(scores: NormativeScores) -> float:
    """Top-two gap on the simplex."""
    ordered = sorted(scores, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0


def dominant_school(scores: NormativeScores) -> str:
    labels = ("consequentialism", "virtue_ethics", "deontology")
    return labels[max(range(3), key=lambda i: scores[i])]


def top_two_ratio(scores: NormativeScores) -> float:
    ordered = sorted(scores, reverse=True)
    if ordered[1] <= 1e-12:
        return float("inf")
    return ordered[0] / ordered[1]


def plurality_features(scores: NormativeScores) -> dict[str, Any]:
    """Eq. (3) style features: margin, entropy, dominant, T1/T2."""
    return {
        "alpha": scores[0],
        "beta": scores[1],
        "gamma": scores[2],
        "dominant_school": dominant_school(scores),
        "margin": score_margin(scores),
        "entropy": normative_entropy(scores),
        "top_two_ratio": top_two_ratio(scores),
        "simplex_ok": simplex_constraint_ok(scores),
    }
