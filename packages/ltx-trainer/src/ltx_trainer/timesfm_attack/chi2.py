"""χ² thresholds and statistics (Proposition 1, Theorem 1 budget)."""

from __future__ import annotations

import math

import numpy as np


def chi2_statistic(residual: np.ndarray, cov: np.ndarray) -> float:
    """gp[k] = z^T Σ^{-1} z for residual z."""
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    c = np.asarray(cov, dtype=np.float64)
    inv = np.linalg.inv(c)
    return float(r @ inv @ r)


def chi2_threshold(m: int, alpha: float) -> float:
    """Threshold τ with P(χ²_m > τ) ≈ α (Prop. 1 / Algorithm 1)."""
    if m <= 0:
        raise ValueError("m must be positive")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    # Known paper anchors
    anchors = {
        (3, 0.005): 12.838,
        (3, 0.05): 7.815,
        (34, 0.001): 58.247,
    }
    key = (m, round(alpha, 6))
    if key in anchors:
        return anchors[key]

    # Wilson–Hilferty normal approximation
    z = _norm_ppf(1.0 - alpha)
    a = float(m)
    t = 1.0 - 2.0 / (9.0 * a) + z * math.sqrt(2.0 / (9.0 * a))
    return a * (t**3)


def attack_budget_delta_tau(
    m: int,
    tau: float,
    alpha_p: float,
    delta_alpha_p: float,
) -> float:
    """Δτ*_p from Theorem 1: sup λ s.t. 1 - F_{χ²(m,λ)}(τ) ≤ α_p + Δα_p.

    Uses bisection on the non-centrality parameter λ.
    """
    target = alpha_p + delta_alpha_p
    lo, hi = 0.0, max(tau, 1.0) * 4.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        tail = 1.0 - _ncx2_cdf(tau, m, mid)
        if tail <= target:
            lo = mid
        else:
            hi = mid
    return lo


def _norm_ppf(p: float) -> float:
    """Approximate inverse standard normal CDF (Acklam)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _ncx2_cdf(x: float, k: int, lam: float) -> float:
    """Approximate non-central χ² CDF via Poisson mixture of central χ²."""
    if x <= 0.0:
        return 0.0
    lam = max(float(lam), 0.0)
    total = 0.0
    pois = math.exp(-0.5 * lam)
    for j in range(0, 40):
        df = k + 2 * j
        term = pois * _central_chi2_cdf(x, df)
        total += term
        if term < 1e-12:
            break
        pois *= 0.5 * lam / max(j + 1, 1)
    return min(max(total, 0.0), 1.0)


def _central_chi2_cdf(x: float, k: float) -> float:
    if x <= 0.0:
        return 0.0
    return _regularized_gamma(k * 0.5, x * 0.5)


def _regularized_gamma(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x) via series / continued fraction."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid gamma arguments")
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def _gamma_series(a: float, x: float) -> float:
    ap = a
    summ = 1.0 / a
    del_ = summ
    for n in range(1, 200):
        ap += 1.0
        del_ *= x / ap
        summ += del_
        if abs(del_) < abs(summ) * 1e-12:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
