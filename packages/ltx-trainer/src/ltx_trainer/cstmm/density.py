"""cSTMM component density and limiting cases — Eqs. (5)–(12)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cstmm.geometry import quadratic_form


def log_density_unnormalized(z: np.ndarray, a: np.ndarray, nu: float, m_channels: int) -> float:
    """Log of Eq. (8) up to normalization: -(ν+M)/2 * log(1 - 2/ν * z^H A z)."""
    q = quadratic_form(z, a)
    inner = 1.0 - (2.0 / nu) * q
    if inner <= 0:
        return -np.inf
    return -0.5 * (nu + m_channels) * float(np.log(inner))


def watson_log_density_unnormalized(z: np.ndarray, a: np.ndarray, kappa: float, nu: float, m_channels: int) -> float:
    """Rank-one case Eq. (12): ∝ [1 + 2κ/ν (1 - |a^H z|^2)]^{-(ν+M)/2}."""
    z = np.asarray(z, dtype=np.complex128).ravel()
    a = np.asarray(a, dtype=np.complex128).ravel()
    a = a / (np.linalg.norm(a) + 1e-12)
    r2 = float(np.abs(a.conj() @ z) ** 2)
    inner = 1.0 + (2.0 * kappa / nu) * (1.0 - r2)
    if inner <= 0:
        return -np.inf
    return -0.5 * (nu + m_channels) * float(np.log(inner))


def limiting_case_label(nu: float, m_channels: int, rank_one: bool = False) -> str:
    """Map ν to included model family (paper §3.1–3.2)."""
    if rank_one and nu >= 1e3:
        return "cWMM (ν → ∞, rank-one)"
    if nu >= 1e3:
        return "cBMM (ν → ∞)"
    if abs(nu - m_channels) < 1e-6:
        return "cACGMM (ν = M)"
    return "cSTMM (general ν)"
