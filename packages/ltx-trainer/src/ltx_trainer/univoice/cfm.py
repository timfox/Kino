"""Conditional flow matching stubs (Eq. 1–2)."""

from __future__ import annotations

import numpy as np


def ot_interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Optimal-transport path ϕ_t(x0, x1) = (1-t)x0 + t x1."""
    return (1.0 - t) * x0 + t * x1


def target_velocity(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Target vector field u = x1 - x0 under OT path."""
    return x1 - x0


def cfm_loss(pred_v: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> float:
    """L_CFM = E || v_θ(ϕ_t) - (x1 - x0) ||^2 (Eq. 2)."""
    u = target_velocity(x0, x1)
    diff = pred_v - u
    return float(np.mean(diff * diff))


def euler_sample_step(x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Single Euler ODE step dx/dt = v."""
    return x + dt * v
