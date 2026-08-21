"""Rectified flow matching loss (Eq. 1–3)."""

from __future__ import annotations

import numpy as np


def interpolate_latent(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Eq. (1): x_t = t x_1 + (1 - t) x_0."""
    return t * x1 + (1.0 - t) * x0


def target_velocity(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Eq. (2): v_t = x_1 - x_0."""
    return x1 - x0


def flow_matching_loss(
    predicted_velocity: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> float:
    """Eq. (3): MSE between u(x_t, c, t) and v_t."""
    vt = target_velocity(x0, x1)
    diff = predicted_velocity - vt
    return float(np.mean(diff**2))
