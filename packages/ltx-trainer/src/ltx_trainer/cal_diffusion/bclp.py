"""BCLP loss structure for co-optimized diffusion + CT blur (Eq. 7–8)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cal_diffusion.kernel import convolve_volume, gaussian_kernel_3d, wiener_deconvolve
from ltx_trainer.cal_diffusion.tomography import blur_operator, ct_blur_volume


def _combined_kernel(shape, *, d_m2_s: float, t_s: float, voxel_m: float) -> np.ndarray:
    k_ct = gaussian_kernel_3d(shape, d_m2_s=2.5e-11, t_s=80.0, voxel_m=voxel_m, truncate_sigma=2.0)
    k_diff = gaussian_kernel_3d(shape, d_m2_s=d_m2_s, t_s=t_s, voxel_m=voxel_m)
    return convolve_volume(k_diff, k_ct)


def bclp_band_error(reconstruction: np.ndarray, target: np.ndarray, *, delta: float = 0.2, eps: float = 0.4) -> np.ndarray:
    """Band-constraint error E(r) from Eq. (7) on in/out regions."""
    mu = float(np.percentile(reconstruction[target > 0.5], 5)) if (target > 0.5).any() else float(reconstruction.mean())
    tl = (1.0 - delta - eps) * mu
    th = (1.0 + delta + eps) * mu
    err = np.zeros_like(reconstruction)
    in_part = target > 0.5
    out_part = ~in_part
    err[in_part] = np.maximum(0.0, np.abs(reconstruction[in_part] - th) - eps * mu)
    err[out_part] = np.maximum(0.0, np.abs(reconstruction[out_part] - tl) - eps * mu)
    return err


def bclp_loss(reconstruction: np.ndarray, target: np.ndarray) -> float:
    """L2 band-constraint norm (Eq. 7)."""
    err = bclp_band_error(reconstruction, target)
    w = 1.0 / np.sqrt(reconstruction.size)
    return float(np.sqrt(np.sum((w * err) ** 2)))


def cooptimize_reconstruction(
    target: np.ndarray,
    *,
    true_d_m2_s: float,
    correction_d_m2_s: float,
    t_s: float,
    voxel_m: float,
    steps: int = 16,
    step_size: float = 0.5,
) -> np.ndarray:
    """Co-optimized dose: Wiener pre-dose + PGD polish on band-constraint loss."""
    k_corr = _combined_kernel(target.shape, d_m2_s=correction_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    reg = 0.015 * (correction_d_m2_s / max(true_d_m2_s, 1e-12)) ** 0.25
    x = wiener_deconvolve(target, k_corr, reg=reg)
    for _ in range(steps):
        y = blur_operator(x, d_m2_s=correction_d_m2_s, t_s=t_s, voxel_m=voxel_m)
        adj = blur_operator(y - target, d_m2_s=correction_d_m2_s, t_s=t_s, voxel_m=voxel_m)
        x -= step_size * adj
        x = np.maximum(x, 0.0)
    return blur_operator(x, d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)


def richardson_lucy_deconv_target(
    target: np.ndarray,
    *,
    d_m2_s: float,
    t_s: float,
    voxel_m: float,
    iterations: int = 12,
) -> np.ndarray:
    """Orth et al. RL pre-compensation — edge over-accentuation (Fig. 7)."""
    k = _combined_kernel(target.shape, d_m2_s=d_m2_s, t_s=t_s, voxel_m=voxel_m)
    estimate = target.copy()
    for _ in range(iterations):
        conv = convolve_volume(estimate, k)
        conv = np.maximum(conv, 1e-8)
        ratio = target / conv
        estimate *= convolve_volume(ratio, k)
        estimate = np.maximum(estimate, 0.0)
    ring = estimate - ct_blur_volume(estimate, voxel_m=voxel_m)
    return np.maximum(estimate + 0.55 * ring, 0.0)
