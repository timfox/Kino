"""Rectified Flow velocity matching and one-step inference (Sec. 2.1, 3.3)."""

from __future__ import annotations

import numpy as np


def velocity_target(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Optimal-transport constant velocity y − x (Eq. 4, 10)."""
    return y - x


def velocity_matching_loss(v_pred: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Lv(θ) = ||(y−x) − vθ||² (Eq. 4)."""
    target = velocity_target(x, y)
    return float(np.mean(np.abs(v_pred - target) ** 2))


def estimate_clean(xt: np.ndarray, t: float, v_pred: np.ndarray, epsilon: float) -> np.ndarray:
    """ˆx = xt − (t − ε)·vθ under linear-path assumption (Sec. 3.3)."""
    return xt - (t - epsilon) * v_pred


def euler_step(xt: np.ndarray, t: float, v_pred: np.ndarray, delta_t: float) -> np.ndarray:
    """ODE update xt−Δt = xt − vθ Δt (Eq. 12)."""
    return xt - v_pred * delta_t


def one_step_enhance(
    y: np.ndarray,
    v_fn,
    *,
    t_start: float = 0.97,
    epsilon: float = 0.03,
) -> np.ndarray:
    """Single-step (NFE=1) enhancement from noisy spectrogram y."""
    delta_t = t_start - epsilon
    v = v_fn(y, t_start, y)
    return euler_step(y, t_start, v, delta_t)
