"""LCMV beamformer baseline (Eq. 13) + helpers."""

from __future__ import annotations

import numpy as np


def lcmv_weights(phi_nn: np.ndarray, c: np.ndarray, g: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Eq. (13): w = Φ_nn^{-1} C (C^H Φ_nn^{-1} C)^{-1} g."""
    phi = np.asarray(phi_nn)
    if phi.ndim != 2 or phi.shape[0] != phi.shape[1]:
        raise ValueError("phi_nn must be square (M, M)")
    c = np.asarray(c)
    g = np.asarray(g).reshape(-1)
    if c.shape[0] != phi.shape[0]:
        raise ValueError("C must be (M, Q) matching phi_nn")
    if c.shape[1] != g.size:
        raise ValueError("g must have length Q (number of constraints)")

    m = phi.shape[0]
    phi_reg = 0.5 * (phi + phi.conj().T) + delta * np.eye(m, dtype=np.complex128)
    inv_phi = np.linalg.inv(phi_reg)
    mid = c.conj().T @ inv_phi @ c
    mid_inv = np.linalg.pinv(mid)
    w = inv_phi @ c @ (mid_inv @ g)
    return w


def wideband_beampower(w_fk: np.ndarray, h_fk_theta: np.ndarray) -> np.ndarray:
    """P(θ)=Σ_k |w^H(k) h(k,θ)|^2 (as used in Sec. 4.2 beampattern analysis)."""
    w = np.asarray(w_fk)  # (K, M)
    h = np.asarray(h_fk_theta)  # (K, M, T)
    if w.ndim != 2:
        raise ValueError("w_fk must be (K, M)")
    if h.ndim != 3 or h.shape[0] != w.shape[0] or h.shape[1] != w.shape[1]:
        raise ValueError("h_fk_theta must be (K, M, T) matching w")
    b = np.einsum("km,kmt->kt", w.conj(), h)  # (K, T)
    return np.sum(np.abs(b) ** 2, axis=0)  # (T,)

