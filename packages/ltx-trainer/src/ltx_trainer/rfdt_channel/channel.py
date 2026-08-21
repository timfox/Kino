"""Multipath CIR and CFR (Eq. 1–2)."""

from __future__ import annotations

import numpy as np


def cir_discrete(
    delays_ns: np.ndarray,
    gains: np.ndarray,
    tau_grid_ns: np.ndarray,
) -> np.ndarray:
    """h(τ) = Σ a_ℓ δ(τ − τ_ℓ) sampled on tau_grid."""
    h = np.zeros_like(tau_grid_ns, dtype=np.complex128)
    for tau_l, a_l in zip(delays_ns, gains):
        idx = int(np.argmin(np.abs(tau_grid_ns - tau_l)))
        h[idx] += a_l
    return h


def cfr_from_paths(
    delays_ns: np.ndarray,
    gains: np.ndarray,
    freq_hz: np.ndarray,
) -> np.ndarray:
    """H(f) = Σ a_ℓ exp(−j2πf τ_ℓ), τ in seconds."""
    tau_s = delays_ns * 1e-9
    h = np.zeros_like(freq_hz, dtype=np.complex128)
    for tau_l, a_l in zip(tau_s, gains):
        h += a_l * np.exp(-1j * 2 * np.pi * freq_hz * tau_l)
    return h


def effective_path_count(gains: np.ndarray, threshold_ratio: float = 0.01) -> int:
    """Count paths above threshold_ratio × max |gain|."""
    if len(gains) == 0:
        return 0
    peak = np.max(np.abs(gains))
    if peak <= 0:
        return 0
    return int(np.sum(np.abs(gains) >= threshold_ratio * peak))
