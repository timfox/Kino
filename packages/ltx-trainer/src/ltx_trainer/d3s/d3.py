"""Dual Differential Defocus depth (Eq. 10, simplified Eq. 17)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.d3s.calibration import a_of_delta_x, b_of_delta_x
from ltx_trainer.d3s.derivatives import gradient_xy, laplacian, window_sum


def d3_depth_eq17(
    i0: np.ndarray,
    i1_shifted: np.ndarray,
    *,
    delta_x_mm: float,
    window_radius: int = 2,
) -> np.ndarray:
    """Per-pixel D3 depth via least-squares window (Eq. 17)."""
    i_dot = i0 - i1_shifted
    i_avg = 0.5 * (i0 + i1_shifted)
    gx, gy = gradient_xy(i_avg)
    ix_dot = gx * i_dot
    num = window_sum(ix_dot, window_radius)
    den = window_sum(i_dot * i_dot, window_radius)
    a = a_of_delta_x(delta_x_mm)
    b = b_of_delta_x(delta_x_mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = a * num / np.maximum(np.abs(den), 1e-12) + b
    z[~np.isfinite(z)] = np.nan
    return z


def d3_depth_eq10_per_pixel(
    i_dot: float,
    gx: float,
    gy: float,
    lap: float,
    *,
    s0_m: float,
    rho: float,
    alpha: float,
) -> float:
    """Closed-form D3 (Eq. 10) for a single pixel — simulation path."""
    inv_s = 1.0 / s0_m if s0_m > 0 else 0.0
    denom = i_dot - alpha * (rho - inv_s) * lap
    if abs(denom) < 1e-12:
        return float("nan")
    # Lateral translation term omitted in snapshot pair stub; use gradient magnitude proxy.
    numer = s0_m * (gx * gx + gy * gy) * 1e-3 - alpha * lap
    return numer / denom


def finite_difference_fields(i0: np.ndarray, i1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. 11 approximations on an aligned pair."""
    i_dot = i0 - i1
    i_avg = 0.5 * (i0 + i1)
    gx, gy = gradient_xy(i_avg)
    lap = laplacian(i_avg)
    return i_dot, gx, gy, lap, i_avg
