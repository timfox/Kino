"""Spearman correlation and TRUST-Pathos helpers (arXiv:2605.22732)."""

from __future__ import annotations

import numpy as np


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (toy, no tie correction)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(a, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    return ranks


def trust_pathos_in_range(score: int) -> bool:
    """TRUST-Pathos integer scale {-2, …, +2}."""
    return -2 <= score <= 2
