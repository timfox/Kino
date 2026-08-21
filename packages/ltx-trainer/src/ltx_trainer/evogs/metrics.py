"""Quality and size metrics (PSNR, SSIM, ghost splats, storage)."""

from __future__ import annotations

import numpy as np


def psnr(gt: np.ndarray, pred: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((peak**2) / mse)


def ssim_proxy(gt: np.ndarray, pred: np.ndarray) -> float:
    """Window-free SSIM proxy for smoke tests."""
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    mu_x, mu_y = gt.mean(), pred.mean()
    var_x, var_y = gt.var(), pred.var()
    cov = float(np.mean((gt - mu_x) * (pred - mu_y)))
    dynamic = float(max(gt.max() - gt.min(), pred.max() - pred.min(), 1e-3))
    c1, c2 = (0.01 * dynamic) ** 2, (0.03 * dynamic) ** 2
    if var_x < 1e-10:
        mse = float(np.mean((gt - pred) ** 2))
        peak = float(max(abs(mu_x), 1e-3))
        return float(max(0.0, 1.0 - mse / (peak**2 + 1e-12)))
    if var_y < 1e-10:
        mse = float(np.mean((gt - pred) ** 2))
        peak = float(max(abs(mu_y), 1e-3))
        return float(max(0.0, 1.0 - mse / (peak**2 + 1e-12)))
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / (den + 1e-12))


def lpips_proxy(gt: np.ndarray, pred: np.ndarray) -> float:
    diff = np.abs(gt - pred)
    return float(0.3 * diff.mean() + 0.7 * diff.std())


def l1_dssim_loss(gt: np.ndarray, pred: np.ndarray, lambda_dssim: float = 0.2) -> float:
    l1 = float(np.mean(np.abs(gt - pred)))
    dssim = 1.0 - ssim_proxy(gt, pred)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


def ghost_splat_ratio(opacities: np.ndarray, threshold: float = 0.005) -> float:
    if opacities.size == 0:
        return 0.0
    return float(np.mean(opacities < threshold))
