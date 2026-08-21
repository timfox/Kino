"""Baseline-dependent D3 parameters a(ΔX), b(ΔX) (Eq. 17, supplement S3.3)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.d3s.config import D3SOptics

# Cubic fit coefficients from supplement Fig. S3 (R² > 0.98); stub anchors.
_A_COEFFS = np.array([-1.2e3, 8.5e2, -1.8e2, 4.2e1], dtype=np.float64)
_B_COEFFS = np.array([2.1e2, -1.4e2, 3.1e1, -0.05], dtype=np.float64)


def poly_eval(coeffs: np.ndarray, x_mm: float) -> float:
    return float(np.polyval(coeffs, x_mm))


def a_of_delta_x(delta_x_mm: float) -> float:
    """Scale multiplier a(ΔX) in Eq. 17."""
    return poly_eval(_A_COEFFS, delta_x_mm)


def b_of_delta_x(delta_x_mm: float) -> float:
    """Depth offset b(ΔX) in Eq. 17."""
    return poly_eval(_B_COEFFS, delta_x_mm)


def shift_for_virtual_baseline(
    z_m: float,
    delta_x_mm: float,
    optics: D3SOptics,
) -> float:
    """Horizontal pixel shift from stereo Eq. 12 (rectified pair)."""
    p_m = optics.pixel_pitch_um * 1e-6
    s0_m = optics.sensor_distance_mm * 1e-3
    baseline_m = optics.baseline_mm * 1e-3
    delta_x_m = delta_x_mm * 1e-3
    effective_baseline = baseline_m - delta_x_m
    return s0_m * effective_baseline / (max(z_m, 1e-6) * p_m)


def shift_image(img: np.ndarray, dx_px: float) -> np.ndarray:
    """Sub-pixel horizontal shift via linear interpolation."""
    h, w = img.shape
    xs = np.arange(w, dtype=np.float64)
    src_x = xs - dx_px
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(h):
        out[y] = np.interp(xs, src_x, img[y], left=img[y, 0], right=img[y, -1])
    return out
