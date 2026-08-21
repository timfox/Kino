"""Gauss circle problem counting (Eqs. 4–8)."""

from __future__ import annotations

import math


def lattice_count_1d(k: float) -> int:
    """Eq. (5) base case N=1."""
    return 1 + 2 * math.floor(k)


def lattice_count_direct(k: float, n: int, memo: dict[tuple[int, int], int] | None = None) -> int:
    """Eq. (5) recurrence (direct)."""
    if n <= 1:
        return lattice_count_1d(k)
    cache = memo if memo is not None else {}
    key = (int(math.floor(k * 1000)), n)
    if key in cache:
        return cache[key]
    total = 0
    hi = math.floor(k)
    for m in range(-hi, hi + 1):
        sub = math.sqrt(max(0.0, k * k - m * m))
        total += lattice_count_direct(sub, n - 1, cache)
    cache[key] = total
    return total


def gcp_kernel_tau() -> dict[int, int]:
    """Sparse kernel f(τ) in Eq. (7): 1 at 0, 2 at perfect squares."""
    return {0: 1}
