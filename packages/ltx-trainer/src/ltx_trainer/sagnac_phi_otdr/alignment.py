"""Cross-correlation spatiotemporal alignment (Eq. 4)."""

from __future__ import annotations

import numpy as np


def cross_correlation(xs: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation R_sp(τ) over valid lags."""
    xs = xs - xs.mean()
    xp = xp - xp.mean()
    n = len(xs)
    out = np.correlate(xs, xp, mode="full")
    norm = np.sqrt(np.sum(xs**2) * np.sum(xp**2)) + 1e-8
    return out / norm


def optimal_lag(xs: np.ndarray, xp: np.ndarray, max_shift: int | None = None) -> tuple[int, float]:
    """Return lag τ maximizing R_sp(τ) with xs(t) aligned to xp(t+τ) (paper Eq. 4)."""
    n = min(len(xs), len(xp))
    limit = max_shift if max_shift is not None else max(1, n // 4)
    best_lag = 0
    best_score = -1.0
    for lag in range(-limit, limit + 1):
        if lag >= 0:
            a, b = xs[lag:n], xp[: n - lag]
        else:
            a, b = xs[: n + lag], xp[-lag:n]
        if len(a) < 2:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-8)
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag, best_score
