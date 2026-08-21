"""Summation bounds g, ḡ for GCP-ISM (Appendix Eqs. 23–24)."""

from __future__ import annotations

import math


def upper_bound_g(x: float, y: float) -> int:
    """Eq. (23) — arg max m with m + (-1)^m y <= x."""
    even = 2 * math.floor((x - y) / 2)
    odd = 2 * math.floor((x + y - 1) / 2) + 1
    return int(max(even, odd))


def lower_bound_g(x: float, y: float) -> int:
    """Eq. (24) — arg min m with m + (-1)^m y >= x."""
    even = 2 * math.ceil((x - y) / 2)
    odd = 2 * math.ceil((x + y - 1) / 2) + 1
    return int(min(even, odd))


def u_distance(m: int, ell_n: float, s_n: float, r_n: float) -> float:
    """Signed distance along dimension N (Eq. 10)."""
    return ell_n * m + ((-1) ** m) * s_n - r_n


def summation_bounds(sqrt_q: float, ell_n: float, s_n: float, r_n: float) -> tuple[int, int]:
    """Eq. (11) bounds a(q), b(q)."""
    bq = upper_bound_g((r_n + sqrt_q) / ell_n, s_n / ell_n)
    aq = lower_bound_g((r_n - sqrt_q) / ell_n, s_n / ell_n)
    return aq, bq
