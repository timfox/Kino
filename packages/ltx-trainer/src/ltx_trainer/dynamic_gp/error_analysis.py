"""Functional L2 error decomposition — Sec. 5, Eq. (5.14)."""

from __future__ import annotations

from typing import Any

import numpy as np


def functional_l2_error(f_true: np.ndarray, f_hat: np.ndarray, dx: float) -> float:
    """∥f − ˆf∥_2 on grid."""
    diff = f_true - f_hat
    return float(np.sqrt(np.sum(diff * diff) * dx))


def error_decomposition_stub(
    psi_inf: np.ndarray,
    p_inf: np.ndarray,
    lam_u: np.ndarray,
    out_of_subspace: float,
    d: int = 1,
) -> dict[str, float]:
    """Three-way steady-state decomposition — Eq. (5.14)."""
    weight = np.kron(np.eye(d), lam_u)
    noise_limited = float(np.trace(weight @ psi_inf))
    leakage_gap = float(np.trace(weight @ (p_inf - psi_inf)))
    return {
        "noise_limited": noise_limited,
        "leakage_gap": max(leakage_gap, 0.0),
        "out_of_subspace": out_of_subspace,
        "total": noise_limited + max(leakage_gap, 0.0) + out_of_subspace,
    }


def error_card() -> dict[str, Any]:
    return {
        "identity": "E[∥f_t − ˆf_t|t∥²_L2] = in-subspace + out-of-subspace (Prop. 5.5)",
        "decomposition": "noise-limited + leakage gap + out-of-subspace (Eq. 5.14)",
        "convergence": "all terms → 0 as M → ∞ (Prop. 5.7)",
    }
