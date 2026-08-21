"""Conditional flow-matching objectives — §3.1.2, Eq. (3)."""

from __future__ import annotations

import numpy as np


def ot_velocity(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Ground-truth OT velocity v = x0 - x1 — Eq. (2)."""
    return np.asarray(x0, dtype=np.float64) - np.asarray(x1, dtype=np.float64)


def interpolate_path(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """xt = (1 - t) x1 + t x0 along straight-line OT path."""
    return (1.0 - t) * x1 + t * x0


def cfm_loss(
    u_pred: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> float:
    """L_CFM = E ||u_theta - (x0 - x1)||^2 — Eq. (3)."""
    target = ot_velocity(x0, x1)
    diff = np.asarray(u_pred, dtype=np.float64) - target
    return float(np.mean(diff**2))


def toy_velocity_predictor(
    xt: np.ndarray,
    t: float,
    timbre: np.ndarray | None = None,
    style: np.ndarray | None = None,
) -> np.ndarray:
    """Linear stand-in for u_theta(xt, t, y, c)."""
    xt = np.asarray(xt, dtype=np.float64)
    bias = np.zeros_like(xt)
    if timbre is not None:
        bias = bias + 0.1 * np.asarray(timbre, dtype=np.float64)[: xt.size]
    if style is not None:
        bias = bias + 0.1 * np.asarray(style, dtype=np.float64)[: xt.size]
    return ot_velocity(xt + bias, np.zeros_like(xt)) * (1.0 - 0.5 * t)
