"""Discrete-time LTV LQR via Riccati recursion (Sec. 4.4)."""

from __future__ import annotations

import numpy as np


def solve_ltv_lqr(
    a_list: list[np.ndarray],
    b_list: list[np.ndarray],
    *,
    q: float | np.ndarray = 1.0,
    r: float | np.ndarray = 1.0,
    q_terminal: float | np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return feedback gains K_s with u = -K z for regulation to origin.

    Dynamics: z_{s+1} = A_s z_s + B_s u_s for s = 0..H-2.
    """
    h = len(a_list)
    if h == 0:
        return []
    if len(b_list) != h:
        raise ValueError("a_list and b_list must have equal length")

    d_lat = a_list[0].shape[0]
    d_u = b_list[0].shape[1]
    q_h = np.eye(d_lat, dtype=np.float64) * (q_terminal if q_terminal is not None else q)
    if np.ndim(q) == 0:
        q_mat = np.eye(d_lat, dtype=np.float64) * float(q)
    else:
        q_mat = np.asarray(q, dtype=np.float64)
    if np.ndim(r) == 0:
        r_mat = np.eye(d_u, dtype=np.float64) * float(r)
    else:
        r_mat = np.asarray(r, dtype=np.float64)

    p = q_h.copy()
    gains: list[np.ndarray] = [np.zeros((d_u, d_lat), dtype=np.float64) for _ in range(h)]

    for s in range(h - 1, -1, -1):
        a = a_list[s]
        b = b_list[s]
        s_inv = np.linalg.inv(r_mat + b.T @ p @ b)
        k = s_inv @ (b.T @ p @ a)
        gains[s] = k
        a_cl = a - b @ k
        p = q_mat + a.T @ p @ a_cl
    return gains


def la_lqr_text_control(
    k_gain: np.ndarray,
    vz: np.ndarray,
    alpha: float,
    *,
    u_bar: np.ndarray | None = None,
) -> np.ndarray:
    """u* = u_bar + K α v_z — Eq. (20) with δz = -α v_z."""
    u_bar = np.zeros(k_gain.shape[0], dtype=np.float64) if u_bar is None else np.asarray(u_bar, dtype=np.float64)
    direction = k_gain @ vz
    return u_bar + alpha * direction
