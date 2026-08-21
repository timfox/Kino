"""Portability metrics (Sec. 3.4)."""

from __future__ import annotations

import math


def transfer_velocity_hours(t0: float, n_tools: int, beta: float) -> float:
    """Tv(n) = T0 * n^{-beta} (Eq. 1)."""
    if n_tools <= 0 or t0 <= 0:
        return float("inf")
    return t0 * (n_tools ** (-beta))


def cross_domain_quality_binary(validated: bool) -> int:
    """Qd(d) ∈ {0,1} (Eq. 2)."""
    return 1 if validated else 0


def orchestration_overhead(t_coord: float, t_prod: float) -> float:
    """Oh = t_coord / (t_coord + t_prod) (Eq. 3)."""
    total = t_coord + t_prod
    if total <= 0:
        return 0.0
    return t_coord / total


def coverage_breadth(quality_flags: dict[str, int]) -> int:
    """Cb = |{d : Qd(d)=1}| (Eq. 4)."""
    return sum(1 for v in quality_flags.values() if v == 1)


def overhead_decomposition(
    t_human_gate: float,
    t_prod: float,
    *,
    o_terminal_pp: float = 0.0,
    o_auto_pp: float = 0.0,
) -> dict[str, float]:
    """Approximate Oh ≈ Ofloor + Oterminal - Oauto (Eq. 6), illustrative."""
    oh = orchestration_overhead(t_human_gate, t_prod)
    floor = orchestration_overhead(t_human_gate, t_prod)  # baseline
    return {
        "oh": round(oh, 4),
        "o_floor_approx": round(floor, 4),
        "o_terminal_penalty_pp": o_terminal_pp,
        "o_auto_dividend_pp": o_auto,
    }
