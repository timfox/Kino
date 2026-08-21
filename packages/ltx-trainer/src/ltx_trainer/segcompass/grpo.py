"""GRPO group-relative advantages (Appendix B.2, Eq. 10)."""

from __future__ import annotations

import math
from statistics import mean, pstdev


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> list[float]:
    r"""Â_i = (r_i - mean(r)) / std(r) — normalized group advantages."""
    if not rewards:
        return []
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    if sigma < eps:
        return [0.0 for _ in rewards]
    return [(r - mu) / sigma for r in rewards]


def clipped_policy_ratio(
    pi_theta: float,
    pi_old: float,
    advantage: float,
    *,
    epsilon: float = 0.2,
) -> float:
    """min(r·Â, clip(r, 1-ε, 1+ε)·Â) — scalar GRPO surrogate term."""
    if pi_old <= 0:
        return 0.0
    ratio = pi_theta / pi_old
    unclipped = ratio * advantage
    clipped = max(1.0 - epsilon, min(1.0 + epsilon, ratio)) * advantage
    return min(unclipped, clipped)


def kl_penalty(pi_theta: float, pi_ref: float) -> float:
    r"""Unbiased KL estimator (Appendix Eq. 11), scalar toy."""
    if pi_theta <= 0:
        return 0.0
    r = pi_ref / pi_theta
    return r - math.log(r) - 1.0
