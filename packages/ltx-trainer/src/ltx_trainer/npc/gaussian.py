"""Gaussian achievability union and Φ helpers (Sec. V, Eq. 37–38)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Standard normal quantile Φ⁻¹(p); Acklam rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # Peter J. Acklam, 2003
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964792408769e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def transfer_direction_pairs(d: int) -> list[tuple[int, int]]:
    """Ordered pairs ``(i, j)`` with ``0 ≤ i, j ≤ d``, ``i ≠ j`` (Eq. 22)."""
    return [(i, j) for i in range(d + 1) for j in range(d + 1) if i != j]


def transfer_tail_prob(n: int, N: int, variance: float, c_scale: float | None = None) -> float:
    """``Φ(-√n / (N √V))`` with optional ``c`` via ``N ≈ c√n`` (Eq. 39)."""
    if variance <= 0 or N <= 0 or n <= 0:
        return 1.0
    thresh = math.sqrt(n) / (N * math.sqrt(variance))
    if c_scale is not None and c_scale > 0:
        thresh = 1.0 / (c_scale * math.sqrt(variance))
    return normal_cdf(-thresh)


def average_gaussian_union(
    c: float,
    variance_fn: Callable[[np.ndarray, int, int], float],
    sample_points: Iterable[np.ndarray],
    d: int,
) -> float:
    """Discrete ``A_W(c)`` from Eq. (37) over finitely many lattice / grid points."""
    pts = list(sample_points)
    if not pts:
        return 1.0
    total = 0.0
    for u in pts:
        u = np.asarray(u, dtype=np.float64)
        s = 0.0
        for i, j in transfer_direction_pairs(d):
            v = max(variance_fn(u, i, j), 1e-15)
            s += normal_cdf(-1.0 / (c * math.sqrt(v)))
        total += s
    return total / len(pts)


def solve_c_epsilon(
    target_eps: float,
    variance_fn: Callable[[np.ndarray, int, int], float],
    sample_points: Iterable[np.ndarray],
    d: int,
    *,
    c_hi: float = 50.0,
    tol: float = 1e-4,
) -> float:
    """``c_ϵ = sup{c : A_W(c) < ϵ}`` via bisection (Eq. 38)."""
    if not 0.0 < target_eps < 1.0:
        raise ValueError("target_eps must be in (0, 1)")
    pts = list(sample_points)
    lo, hi = 1e-6, c_hi
    if average_gaussian_union(lo, variance_fn, pts, d) >= target_eps:
        return lo
    while average_gaussian_union(hi, variance_fn, pts, d) > target_eps and hi < 1e6:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if average_gaussian_union(mid, variance_fn, pts, d) < target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def bsc_variance(q: float) -> float:
    """Local variance proxy ``q(1-q)`` on reachable interval (Sec. VI-A)."""
    return max(q * (1.0 - q), 1e-15)


def bsc_average_union(c: float, delta: float, *, grid: int = 200) -> float:
    """``A_W(c)`` for BSC with ``λ* = 1-2δ`` (Sec. VI-A integral, discrete)."""
    if not 0.0 < delta < 0.5:
        raise ValueError("delta in (0, 0.5)")
    qs = np.linspace(delta, 1.0 - delta, grid)
    scale = 1.0 - 2.0 * delta
    total = 0.0
    for q in qs:
        v = bsc_variance(q)
        total += 2.0 * normal_cdf(-scale / (2.0 * c * math.sqrt(v)))
    return total / len(qs)


def solve_bsc_c_epsilon(delta: float, epsilon: float) -> float:
    """Solve ``c_{δ,ϵ}`` from Sec. VI-A (``c_ϵ`` with ``1-2δ`` factor in paper)."""
    if not 0.0 < delta < 0.5:
        raise ValueError("delta in (0, 0.5)")

    def aw(c: float) -> float:
        return bsc_average_union(c, delta)

    lo, hi = 1e-6, 100.0
    while aw(hi) > epsilon and hi < 1e6:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if aw(mid) < epsilon:
            lo = mid
        else:
            hi = mid
    return lo


def channel_3x4_variance(u: np.ndarray, i: int, j: int) -> float:
    from ltx_trainer.npc.geometry import channel_3x4_transfer_variance

    return channel_3x4_transfer_variance(u, i, j)


def lattice_gaussian_average(
    n: int,
    N: int,
    lattice_points: Sequence[np.ndarray],
    variance_fn: Callable[[np.ndarray, int, int], float],
    d: int,
) -> float:
    """Finite-lattice ``A^{lat}_{n,N}`` from Eq. (42)."""
    if not lattice_points:
        return 1.0
    total = 0.0
    for u in lattice_points:
        s = 0.0
        for i, j in transfer_direction_pairs(d):
            v = max(variance_fn(np.asarray(u), i, j), 1e-15)
            s += normal_cdf(-math.sqrt(n) / (N * math.sqrt(v)))
        total += s
    return total / len(lattice_points)
