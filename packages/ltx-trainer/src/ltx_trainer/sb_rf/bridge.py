"""Schrödinger Bridge and Brownian Bridge marginals (Sec. 2.2, 3.1–3.2)."""

from __future__ import annotations

import numpy as np


def sb_marginal_weights(t: float, alpha_t: float, sigma_t: float, sigma_T: float) -> tuple[float, float, float]:
    """Eq. (8): wx(t), wy(t), and variance σ²_x(t) for Gaussian SB."""
    alpha_bar_t = alpha_t / max(alpha_t, 1e-8)  # caller supplies schedule; placeholder unity α
    sigma_bar_sq = max(sigma_T**2 - sigma_t**2, 0.0)
    denom = max(sigma_T**2, 1e-8)
    wx = alpha_t * sigma_bar_sq / denom
    wy = alpha_bar_t * (sigma_t**2) / denom
    var_x = (alpha_t**2) * sigma_bar_sq * (sigma_t**2) / denom
    return wx, wy, var_x


def sb_sample_xt(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    rng: np.random.Generator,
    *,
    wx: float | None = None,
    wy: float | None = None,
    var_x: float = 0.0,
) -> np.ndarray:
    """Sample perturbed spectrogram xt from SB marginal (Eq. 7–8)."""
    if wx is None or wy is None:
        wx, wy, var_x = sb_marginal_weights(t, alpha_t=1.0, sigma_t=t, sigma_T=1.0)
    mean = wx * x + wy * y
    if var_x <= 0.0:
        return mean
    noise = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    noise = noise / np.sqrt(2.0)
    return mean + np.sqrt(var_x) * noise


def bb_sample_xt(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    rng: np.random.Generator,
    *,
    sigma_t: float = 0.1,
) -> np.ndarray:
    """Brownian-bridge perturbation (Eq. 9)."""
    z = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    z = z / np.sqrt(2.0)
    return (1.0 - t) * x + t * y + sigma_t * z


def rf_linear_path(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Deterministic RF interpolation (Eq. 3)."""
    return t * y + (1.0 - t) * x
