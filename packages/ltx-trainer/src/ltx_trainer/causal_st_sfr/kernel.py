"""Stochastic wave-equation prior: κ(Δ), discretized C(r,r';ℓ), far-field diffuse coherence."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

_TWO_PI: Final[float] = 2.0 * math.pi


def kappa_band_limited(delta_s: float | np.ndarray, *, omega1: float, omega2: float) -> float | np.ndarray:
    """Eq. (15): stationary temporal covariance from flat PSD on [ω1, ω2] ∪ negatives; κ(0)=1."""
    if omega2 <= omega1:
        raise ValueError("require omega2 > omega1")
    denom_w = omega2 - omega1
    d = np.asarray(delta_s, dtype=np.float64)
    if d.ndim == 0:
        return 1.0
    out = np.ones_like(d, dtype=np.float64)
    mask = np.abs(d) > 1e-15
    out[mask] = (np.sin(omega2 * d[mask]) - np.sin(omega1 * d[mask])) / (d[mask] * denom_w)
    return out


def discrete_covariance_C(
    r_a: np.ndarray,
    r_b: np.ndarray,
    lag_l: int,
    *,
    ts_s: float,
    c_sound: float,
    q: float,
    sphere_pts: np.ndarray,
    omega1: float,
    omega2: float,
) -> np.ndarray:
    """Eq. (18): quadrature approximation of C(r_a, r_b; ℓ) for vectors r_a (P,3), r_b (R,3).

    Returns shape (P, R).
    """
    qn = int(sphere_pts.shape[0])
    if qn < 1:
        raise ValueError("sphere_pts must be non-empty")
    area_weight = 4.0 * math.pi * (float(np.mean(np.linalg.norm(sphere_pts, axis=1))) ** 2) / qn
    # sphere_pts are on sphere radius a; use exact 4πa²/Q if all radii equal
    radii = np.linalg.norm(sphere_pts, axis=1)
    a = float(np.mean(radii)) if radii.size else 1.0
    area_weight = 4.0 * math.pi * (a**2) / qn

    # Pairwise geometry: (P,1,3) - (1,Q,3)
    v_ap = r_a[:, None, :] - sphere_pts[None, :, :]
    v_bp = r_b[:, None, :] - sphere_pts[None, :, :]
    d_ap = np.linalg.norm(v_ap, axis=2)  # (P, Q)
    d_bp = np.linalg.norm(v_bp, axis=2)  # (R, Q)
    # Broadcast to (P, R, Q)
    d1 = d_ap[:, None, :]
    d2 = d_bp[None, :, :]
    delay_arg = lag_l * ts_s - (d1 + d2) / c_sound
    kvals = np.asarray(kappa_band_limited(delay_arg, omega1=omega1, omega2=omega2), dtype=np.float64)
    integrand = (1.0 / (d1 * d2)) * kvals * area_weight
    coeff = q / (16.0 * math.pi**2)
    c_mat = coeff * np.sum(integrand, axis=2)
    # Quadrature with oscillatory κ can yield tiny negative self-variances; clamp for PSD stubs.
    if c_mat.shape[0] == c_mat.shape[1]:
        diag = np.diag(c_mat)
        np.fill_diagonal(c_mat, np.maximum(diag, 0.0))
    return c_mat


def diffuse_far_field_normalized_coherence(distance_m: float | np.ndarray, *, omega: float, c_sound: float):
    """Eq. (21): γ = sinc((ω/c) d) in the normalized far-field diffuse limit."""
    d = np.asarray(distance_m, dtype=np.float64)
    x = (omega / c_sound) * d
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-12
    xm = x[mask]
    out[mask] = np.sin(xm) / xm
    return out
