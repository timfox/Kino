"""Multi-index sets A_κ and A_proj (Eq. 2.4–2.5)."""

from __future__ import annotations

from itertools import product


def multi_indices(order: int, dims: int = 4) -> list[tuple[int, ...]]:
    """Return all α with |α| ≤ order in N^dims."""
    out: list[tuple[int, ...]] = []
    for total in range(order + 1):
        for parts in product(range(total + 1), repeat=dims):
            if sum(parts) == total:
                out.append(parts)
    return out


def projected_indices(order: int) -> list[tuple[int, int]]:
    """A_proj,κ — Eq. (2.5)."""
    return [(a, b) for a, b in multi_indices(order, dims=2)]


def alpha_size(order: int) -> int:
    return len(multi_indices(order))


def factorial_multinomial(alpha: tuple[int, ...], beta: tuple[int, ...]) -> float:
    """binom(α, β) product — Eq. (3.5)."""
    from math import comb

    out = 1.0
    for a, b in zip(alpha, beta, strict=True):
        out *= comb(a, b)
    return out
