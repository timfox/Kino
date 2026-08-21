"""Finite-dimensional Kalman filter on basis coefficients — Sec. 4.1, Lemma 4.3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanState:
    z: np.ndarray
    psi: np.ndarray


def kalman_update(
    state: KalmanState,
    y: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
) -> KalmanState:
    """DGP update — Eqs. (4.3), (4.6)."""
    s = c @ state.psi @ c.T + w
    gamma = state.psi @ c.T @ np.linalg.pinv(s)
    z = state.z + gamma @ (y - c @ state.z)
    psi = state.psi - gamma @ c @ state.psi
    return KalmanState(z=z, psi=psi)


def kalman_predict(state: KalmanState, a_m: np.ndarray, lam_v: np.ndarray) -> KalmanState:
    """DGP prediction — Eqs. (4.4), (4.7)."""
    z = a_m @ state.z
    psi = a_m @ state.psi @ a_m.T + lam_v
    return KalmanState(z=z, psi=psi)


def mean_on_grid(u_grid: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ˆf_t|t(x_i) = U(x_i)^T z for scalar D=1."""
    return u_grid.T @ z[: u_grid.shape[0]]
