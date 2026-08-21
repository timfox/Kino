"""Steering vectors and subspace projectors — Eqs. (1)–(2)."""

from __future__ import annotations

import numpy as np


def normalize_observation(ez: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """z = ez / ||ez||_2 for multichannel STFT bin."""
    ez = np.asarray(ez, dtype=np.complex128).ravel()
    n = float(np.linalg.norm(ez))
    if n < eps:
        return None
    return ez / n


def steering_vector_planar(
    mic_xy: np.ndarray,
    source_xy: np.ndarray,
    frequency_hz: float,
    c: float = 343.0,
) -> np.ndarray:
    """Toy planar-wave steering vector h_f(x) ∈ C^M."""
    mic_xy = np.asarray(mic_xy, dtype=np.float64)
    source_xy = np.asarray(source_xy, dtype=np.float64).ravel()[:2]
    dist = np.linalg.norm(mic_xy - source_xy, axis=1)
    phase = -2.0 * np.pi * frequency_hz * dist / c
    return np.exp(1j * phase)


def mixing_matrix(
    mic_xy: np.ndarray,
    positions: list[np.ndarray],
    frequency_hz: float,
    c: float = 343.0,
) -> np.ndarray:
    """H_f ∈ C^{M×N_valid} from valid target positions."""
    cols = [steering_vector_planar(mic_xy, p, frequency_hz, c) for p in positions]
    if not cols:
        return np.zeros((mic_xy.shape[0], 0), dtype=np.complex128)
    return np.column_stack(cols)


def subspace_projector(h: np.ndarray) -> np.ndarray:
    """P_f = H (H^H H)^{-1} H^H — Eq. (2); Moore–Penrose if rank-deficient."""
    if h.size == 0 or h.shape[1] == 0:
        return np.zeros((h.shape[0], h.shape[0]), dtype=np.complex128)
    hh = h.conj().T @ h
    inv = np.linalg.pinv(hh)
    return h @ inv @ h.conj().T
