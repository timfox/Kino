"""Multi-precision digit utilities (base-B limb arrays)."""

from __future__ import annotations


def precision_digits(u: int, base: int) -> int:
    """Return h with base^{h-1} < u ≤ base^h (paper ``prec(u)`` for u > 0)."""
    if u <= 0:
        return 0
    h = 0
    bound = 1
    while bound < u:
        h += 1
        bound *= base
    return h


def shift_n(u: int, n: int, base: int) -> int:
    """Whole shift shift_{n,B}(u) = floor(u · B^n) for integer n (Definition 1)."""
    if n >= 0:
        return u * pow(base, n)
    return u // pow(base, -n)


def shinv_direct(v: int, h: int, base: int) -> int:
    """Exact shifted inverse shinv_{h,B}(v) = floor(B^h / v)."""
    if v <= 0:
        raise ValueError("divisor must be positive")
    return pow(base, h) // v


def to_limbs(u: int, m: int, base: int) -> list[int]:
    """Least-significant digit first, fixed width m."""
    limbs = [0] * m
    x = u
    for i in range(m):
        if x == 0:
            break
        limbs[i] = x % base
        x //= base
    return limbs


def from_limbs(limbs: list[int], base: int) -> int:
    out = 0
    mul = 1
    for d in limbs:
        out += d * mul
        mul *= base
    return out
