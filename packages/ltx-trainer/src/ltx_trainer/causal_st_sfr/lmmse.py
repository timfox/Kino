"""Causal finite-window LMMSE reconstruction (Eqs. 28–29) — toy dimensions only."""

from __future__ import annotations

import numpy as np

from ltx_trainer.causal_st_sfr.kernel import discrete_covariance_C


def build_covariance_blocks(
    mic_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    window_w: int,
    ts_s: float,
    c_sound: float,
    q: float,
    sphere_pts: np.ndarray,
    omega1: float,
    omega2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Kuu (P,P), Kyy (MW,MW), Kuy (P,MW) from discretized C."""
    m = int(mic_xyz.shape[0])
    p = int(target_xyz.shape[0])
    w = int(window_w)
    kuu = discrete_covariance_C(target_xyz, target_xyz, 0, ts_s=ts_s, c_sound=c_sound, q=q, sphere_pts=sphere_pts, omega1=omega1, omega2=omega2)
    kyy = np.zeros((m * w, m * w), dtype=np.float64)
    kuy = np.zeros((p, m * w), dtype=np.float64)
    for wi in range(w):
        for wj in range(w):
            block = discrete_covariance_C(
                mic_xyz,
                mic_xyz,
                wj - wi,
                ts_s=ts_s,
                c_sound=c_sound,
                q=q,
                sphere_pts=sphere_pts,
                omega1=omega1,
                omega2=omega2,
            )
            kyy[wi * m : (wi + 1) * m, wj * m : (wj + 1) * m] = block
    for wi in range(w):
        block = discrete_covariance_C(
            target_xyz,
            mic_xyz,
            wi,
            ts_s=ts_s,
            c_sound=c_sound,
            q=q,
            sphere_pts=sphere_pts,
            omega1=omega1,
            omega2=omega2,
        )
        kuy[:, wi * m : (wi + 1) * m] = block
    return kuu, kyy, kuy


def lmmse_posterior_mean(
    y_vec: np.ndarray,
    kuu: np.ndarray,
    kuy: np.ndarray,
    kyy: np.ndarray,
    *,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (28)–(29): posterior mean û and covariance Σ_{u|y}."""
    n = kyy.shape[0]
    reg = kyy + sigma2 * np.eye(n, dtype=np.float64)
    inv_reg = np.linalg.inv(reg)
    u_hat = kuy @ inv_reg @ y_vec
    sigma_u = kuu - kuy @ inv_reg @ kuy.T
    return u_hat, sigma_u


def trace_posterior_variance_masked(
    kuu: np.ndarray,
    kuy: np.ndarray,
    kyy: np.ndarray,
    mask: np.ndarray,
    *,
    sigma2: float,
) -> float:
    """Eq. (32) trace Σ_{u|ỹ} for binary mask z (subset of MW observations)."""
    idx = np.flatnonzero(mask.astype(bool))
    if idx.size == 0:
        return float(np.trace(kuu))
    kyy_s = kyy[np.ix_(idx, idx)]
    kuy_s = kuy[:, idx]
    reg = kyy_s + sigma2 * np.eye(kyy_s.shape[0], dtype=np.float64)
    inv_reg = np.linalg.inv(reg)
    sigma_u = kuu - kuy_s @ inv_reg @ kuy_s.T
    return float(np.trace(sigma_u))
