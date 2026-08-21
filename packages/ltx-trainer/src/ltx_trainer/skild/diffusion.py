"""SKILD forward marginals and DDPM reverse steps in DCT space."""

from __future__ import annotations

import numpy as np

from ltx_trainer.skild.dct import dct2, idct2
from ltx_trainer.skild.schedule import SkildSchedule


def _broadcast_schedule(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Broadcast ``(H,W)`` schedule terms to match DCT tensor rank."""
    if arr.ndim == target.ndim:
        return arr
    if target.ndim == 3:
        return arr[..., None]
    return arr


def forward_marginal_dct(
    x0_dct: np.ndarray,
    spectrum: np.ndarray,
    schedule: SkildSchedule,
    step: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``X_n`` from forward marginal (Eq. 4). Returns ``(X_n, ε)``."""
    ab = _broadcast_schedule(schedule.alpha_bar(step), x0_dct)
    eps = rng.standard_normal(size=x0_dct.shape).astype(np.float32)
    sqrt_ab = np.sqrt(np.maximum(ab, 0.0)).astype(np.float32)
    sqrt_one_minus = np.sqrt(np.maximum(1.0 - ab, 0.0)).astype(np.float32)
    noise_scale = np.sqrt(np.maximum(spectrum, 1e-12)).astype(np.float32)
    xn = sqrt_ab * x0_dct + sqrt_one_minus * noise_scale * eps
    return xn.astype(np.float32), eps


def reverse_step_dct(
    xn: np.ndarray,
    eps_pred: np.ndarray,
    spectrum: np.ndarray,
    schedule: SkildSchedule,
    step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ancestral DDPM reverse step (Eq. A.17 / A.14) with ε-prediction."""
    alpha = _broadcast_schedule(schedule.alpha_step(step), xn)
    ab = _broadcast_schedule(schedule.alpha_bar(step), xn)
    ab_prev = _broadcast_schedule(schedule.alpha_bar(step - 1), xn)
    beta = _broadcast_schedule(schedule.beta_step(step), xn)

    alpha = np.maximum(alpha, schedule.cfg.alpha_floor)
    ab = np.maximum(ab, schedule.cfg.alpha_floor)
    ab_prev = np.maximum(ab_prev, schedule.cfg.alpha_floor)

    s0_sqrt = np.sqrt(np.maximum(spectrum, 1e-12)).astype(np.float32)
    one_minus_ab = np.maximum(1.0 - ab, 1e-12)
    one_minus_ab_prev = np.maximum(1.0 - ab_prev, 1e-12)

    coef_eps = beta / (s0_sqrt * np.sqrt(one_minus_ab))
    mu = (xn - coef_eps * eps_pred) / np.sqrt(alpha)

    beta_tilde = beta * one_minus_ab_prev / one_minus_ab
    if step <= 0:
        return mu.astype(np.float32)
    z = rng.standard_normal(size=xn.shape).astype(np.float32)
    return (mu + np.sqrt(np.maximum(beta_tilde, 0.0)) * s0_sqrt * z).astype(np.float32)


def image_to_dct(img: np.ndarray) -> np.ndarray:
    """``(H,W)`` or ``(H,W,C)`` in [0,1] -> DCT coeffs same shape."""
    if img.ndim == 2:
        return dct2(img.astype(np.float32))
    out = np.stack([dct2(img[..., c]) for c in range(img.shape[-1])], axis=-1)
    return out.astype(np.float32)


def dct_to_image(coeff: np.ndarray) -> np.ndarray:
    if coeff.ndim == 2:
        return idct2(coeff.astype(np.float64)).astype(np.float32)
    out = np.stack([idct2(coeff[..., c]) for c in range(coeff.shape[-1])], axis=-1)
    return out.astype(np.float32)
