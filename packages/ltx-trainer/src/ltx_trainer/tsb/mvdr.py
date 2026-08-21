"""MVDR / Capon beamformer — §III, Eqs. (4)–(8)."""

from __future__ import annotations

import numpy as np


def mvdr_weights(
    scm: np.ndarray,
    steering: np.ndarray,
    delta: float = 1e-3,
) -> np.ndarray:
    """w = R^{-1} ν / (ν^T R^{-1} ν) with diagonal loading — Eq. (6)/(8)."""
    p = steering.shape[0]
    r = scm + delta * np.eye(p)
    u = np.linalg.solve(r, steering)
    denom = float(steering @ u)
    if abs(denom) < 1e-12:
        return steering / max(float(steering @ steering), 1e-12)
    return u / denom


def sample_covariance(snapshots: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """SCM from p×T snapshot matrix — Eq. (7)."""
    if snapshots.ndim != 2:
        raise ValueError("snapshots must be p×T")
    p, t = snapshots.shape
    if t == 0:
        return delta * np.eye(p)
    s = snapshots @ snapshots.T / t
    return s + delta * np.eye(p)


def segment_output_power(
    snapshots: np.ndarray,
    steering: np.ndarray,
    delta: float = 1e-3,
) -> float:
    """E(i,j) = min_w Σ |w^T x[t]|^2 subject to w^T ν = 1 — Eq. (20)."""
    w = mvdr_weights(sample_covariance(snapshots, delta), steering, delta)
    outputs = w @ snapshots
    return float(np.sum(outputs**2))


def woodbury_inverse_update(
    inv_r: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rank-one inverse update — Algorithm 1 lines 10–12."""
    u = inv_r @ x
    gamma = 1.0 + float(x @ u)
    inv_new = inv_r - np.outer(u, u) / gamma
    return inv_new, u, gamma
