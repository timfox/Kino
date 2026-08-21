"""STRESS metric (García et al., Eqs. 1–2)."""

from __future__ import annotations

import numpy as np


def scaling_factor(delta_e: np.ndarray, delta_v: np.ndarray) -> float:
    """F = ΣΔE² / Σ(ΔE·ΔV)."""
    de = np.asarray(delta_e, dtype=np.float64)
    dv = np.asarray(delta_v, dtype=np.float64)
    denom = float(np.sum(de * dv))
    if abs(denom) < 1e-12:
        return 1.0
    return float(np.sum(de**2) / denom)


def stress(delta_e: np.ndarray, delta_v: np.ndarray) -> float:
    """Standardized Residual Sum of Squares (lower is better)."""
    de = np.asarray(delta_e, dtype=np.float64)
    dv = np.asarray(delta_v, dtype=np.float64)
    f = scaling_factor(de, dv)
    pred = f * dv
    num = float(np.sum((de - pred) ** 2))
    den = float(np.sum(pred**2))
    if den < 1e-12:
        return 0.0
    return 100.0 * float(np.sqrt(num / den))
