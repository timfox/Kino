"""Uncertainty scores and selective-prediction metrics (arXiv:2605.22746)."""

from __future__ import annotations

import numpy as np


def vacuity(alpha: np.ndarray, num_classes: int, shift_prior: bool = False) -> float:
    """u_vacuity = K / α0; optional α_i = e_i + 1 shift when prior mass absent (Eq. 16)."""
    a = np.asarray(alpha, dtype=np.float64)
    if shift_prior:
        a = a + 1.0
    a0 = float(np.sum(a))
    if a0 <= 0:
        return float("inf")
    return float(num_classes / a0)


def normalized_entropy(p_hat: np.ndarray, num_classes: int, eps: float = 1e-12) -> float:
    """u_entropy = −Σ p̂_j log p̂_j / log K (Eq. 16)."""
    p = np.asarray(p_hat, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    ent = -float(np.sum(p * np.log(p)))
    return ent / max(np.log(num_classes), eps)


def selective_prediction_metrics(
    correct: np.ndarray,
    uncertainties: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute Accth, Acctotal, Coverage for uncertainty threshold t (Section 5.2)."""
    correct = np.asarray(correct, dtype=bool)
    u = np.asarray(uncertainties, dtype=np.float64)
    accept = u <= threshold
    nc = int(np.sum(correct & accept))
    nf = int(np.sum((~correct) & accept))
    nw = int(np.sum(~accept))
    denom_th = nc + nf
    denom_all = nc + nf + nw
    accth = nc / denom_th if denom_th else 0.0
    acctotal = nc / denom_all if denom_all else 0.0
    coverage = denom_th / denom_all if denom_all else 0.0
    return {
        "accth": float(accth),
        "acctotal": float(acctotal),
        "coverage": float(coverage),
        "n_correct": float(nc),
        "n_wrong": float(nf),
        "n_withheld": float(nw),
    }
