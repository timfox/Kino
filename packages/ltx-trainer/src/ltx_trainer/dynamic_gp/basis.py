"""Basis functions and separable kernel structure — Sec. 2.1, 4.1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def fourier_basis(x: np.ndarray, m: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """M Fourier modes on [lo, hi] → U(x) ∈ R^M."""
    x = np.asarray(x, dtype=float)
    L = hi - lo
    out = np.zeros((m, x.size))
    for k in range(m):
        if k == 0:
            out[k] = 1.0 / math.sqrt(L)
        else:
            freq = math.pi * k / L
            out[k] = math.sqrt(2.0 / L) * np.cos(freq * (x - lo))
    return out


def gram_matrix(u_at_grid: np.ndarray, dx: float) -> np.ndarray:
    """Λ_U ≈ ∫ U(x)U(x)^T dν(x) via Riemann sum."""
    return u_at_grid @ u_at_grid.T * dx


def separable_kernel(u_x: np.ndarray, u_xp: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Scalar separable kernel k(x,x') = U(x)^T Λ U(x')."""
    return float(u_x @ lam @ u_xp)


def block_basis_eval(u_x: np.ndarray, d: int) -> np.ndarray:
    """ˇU(x) = I_D ⊗ U(x) stacked for vector-valued state."""
    m = u_x.shape[0]
    out = np.zeros((d * m, d))
    for i in range(d):
        out[i * m : (i + 1) * m, i] = u_x
    return out


def transition_matrix(lam_f: np.ndarray, lam_u: np.ndarray, d: int) -> np.ndarray:
    """A_M = Λ (I_D ⊗ Λ_U) — Lemma 4.1."""
    dm = lam_f.shape[0]
    kron = np.kron(np.eye(d), lam_u)
    return lam_f @ kron


def morphology_card() -> dict[str, Any]:
    return {
        "separable": "K(x,x') = ˇU(x)^T Λ ˇU(x')",
        "transition": "A_M = Λ (I_D ⊗ Λ_U)",
        "coefficient_state": "z_t ∈ R^{DM}",
    }
