"""Normalized multichannel observations on the complex unit sphere — Eq. (1)."""

from __future__ import annotations

import numpy as np


def normalize_observation(y: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    """z = y / ||y||_2; return None if below energy threshold."""
    y = np.asarray(y, dtype=np.complex128).ravel()
    norm = float(np.linalg.norm(y))
    if norm < eps:
        return None
    return y / norm


def quadratic_form(z: np.ndarray, a: np.ndarray) -> float:
    """z^H A z for Hermitian A."""
    z = np.asarray(z, dtype=np.complex128).ravel()
    a = np.asarray(a, dtype=np.complex128)
    return float(np.real(z.conj() @ a @ z))


def is_valid_precision(a: np.ndarray, nu: float) -> bool:
    """Constraint λ_max(A) < ν/2 — Eq. (9), canonical λ_max = 0 representation."""
    eigvals = np.linalg.eigvalsh(np.asarray(a, dtype=np.complex128))
    return float(np.max(eigvals)) < 1e-10 and float(nu) > 2.0 * float(np.max(np.abs(eigvals)))
