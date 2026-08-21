"""PSNR, MS-SSIM, and bpp helpers (Sec. IV-C, V)."""

from __future__ import annotations

import math

import numpy as np


def psnr_db(reference: np.ndarray, estimate: np.ndarray, *, peak: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    mse = float(np.mean((ref - est) ** 2))
    if mse <= 1e-16:
        return 99.0
    return 10.0 * math.log10((peak**2) / mse)


def ms_ssim(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Lightweight MS-SSIM proxy (structural similarity mean); not a full Wang implementation."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("shape mismatch")
    c1, c2 = 0.01**2, 0.03**2
    scales = []
    r, e = ref, est
    for _ in range(3):
        mu_r, mu_e = r.mean(), e.mean()
        var_r, var_e = r.var(), e.var()
        cov = ((r - mu_r) * (e - mu_e)).mean()
        ssim = ((2 * mu_r * mu_e + c1) * (2 * cov + c2)) / (
            (mu_r**2 + mu_e**2 + c1) * (var_r + var_e + c2)
        )
        scales.append(float(np.clip(ssim, 0.0, 1.0)))
        if min(r.shape[:2]) < 4:
            break
        r = r[::2, ::2] if r.ndim == 2 else r[::2, ::2, :]
        e = e[::2, ::2] if e.ndim == 2 else e[::2, ::2, :]
    return float(np.mean(scales))


def bpp_from_param_count(
    n_params: int,
    *,
    n_frames: int,
    height: int,
    width: int,
    bits_per_weight: int = 32,
) -> float:
    """Bits per pixel from weight storage (INR-style representation)."""
    total_bits = n_params * bits_per_weight
    pixels = max(1, n_frames * height * width)
    return total_bits / pixels
