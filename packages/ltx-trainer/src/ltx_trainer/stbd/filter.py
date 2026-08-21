"""Toy PF helpers: MMSE estimate, boundary factor, RMSE — Eqs. (6)–(7)."""

from __future__ import annotations

import numpy as np


def nearly_constant_velocity_step(
    state: np.ndarray,
    dt: float,
    q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """4D [px, py, vx, vy] constant-velocity model — Eq. (5)."""
    px, py, vx, vy = state
    ax = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    bx = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]], dtype=np.float64)
    noise = rng.normal(0.0, q, size=2)
    return ax @ np.array([px, py, vx, vy]) + bx @ noise


def boundary_factor(position_xy: np.ndarray, room_size: float, tau: float) -> float:
    """exp(-d^2/τ^2) when outside [0, room_size]^2."""
    x, y = float(position_xy[0]), float(position_xy[1])
    d = 0.0
    if x < 0:
        d = max(d, -x)
    elif x > room_size:
        d = max(d, x - room_size)
    if y < 0:
        d = max(d, -y)
    elif y > room_size:
        d = max(d, y - room_size)
    if d <= 0:
        return 1.0
    return float(np.exp(-(d**2) / (tau**2)))


def mmse_position(
    particles: np.ndarray,
    weights: np.ndarray,
    slot: int,
) -> np.ndarray:
    """Minimum mean square error position for slot n — Eq. (6)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    states = particles[:, slot * 4 : slot * 4 + 2]
    return np.average(states, axis=0, weights=w)


def position_rmse(
    true_positions: list[tuple[int, int, np.ndarray]],
    est_positions: list[tuple[int, int, np.ndarray]],
) -> float:
    """RMSE over valid (t, n) pairs — Eq. (7)."""
    if not true_positions:
        return 0.0
    errs = []
    est_map = {(t, n): p for t, n, p in est_positions}
    for t, n, p_true in true_positions:
        p_est = est_map.get((t, n))
        if p_est is not None:
            errs.append(float(np.linalg.norm(p_est - p_true) ** 2))
    if not errs:
        return float("inf")
    return float(np.sqrt(np.mean(errs)))
