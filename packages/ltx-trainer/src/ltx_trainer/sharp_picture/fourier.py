"""Fourier–Walsh basis on Boolean domains (Eq. 1)."""

from __future__ import annotations

from typing import Iterable


def parity_characteristic(subset: Iterable[int], x: list[int] | tuple[int, ...]) -> float:
    """χ_A(x) = (1 + (-1)^{Σ_{i∈A} x_i}) / 2 for x_i ∈ {0, 1}."""
    s = sum(x[i] for i in subset)
    return (1.0 + (-1) ** s) / 2.0


def fourier_expansion(
    coefficients: dict[frozenset[int], float],
    x: list[int] | tuple[int, ...],
) -> float:
    """f(x) = Σ_A λ_A χ_A(x)."""
    return sum(c * parity_characteristic(A, x) for A, c in coefficients.items())


def random_positive_coefficients(
    sparsity: int,
    degree: int,
    context_length: int,
    rng=None,
) -> dict[frozenset[int], float]:
    """Sparse spectrum with |A| = degree and positive λ_A (Sec. 2.1)."""
    import random

    rng = rng or random.Random(0)
    coeffs: dict[frozenset[int], float] = {}
    seen: set[frozenset[int]] = set()
    for _ in range(sparsity):
        for _attempt in range(100):
            indices = tuple(sorted(rng.sample(range(context_length), degree)))
            key = frozenset(indices)
            if key not in seen:
                seen.add(key)
                coeffs[key] = abs(rng.gauss(0, 1))
                break
    total = sum(c * c for c in coeffs.values()) or 1.0
    scale = 1.0 / (total ** 0.5)
    return {k: v * scale for k, v in coeffs.items()}


def degree_of_spectrum(coefficients: dict[frozenset[int], float]) -> int:
    if not coefficients:
        return 0
    return max(len(A) for A in coefficients)


def sparsity_of_spectrum(coefficients: dict[frozenset[int], float]) -> int:
    return len(coefficients)