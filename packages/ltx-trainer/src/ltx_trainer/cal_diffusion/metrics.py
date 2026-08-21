"""ASSD, MS-SSIM, and paper table anchors (Figs 3, 6, 7)."""

from __future__ import annotations

from typing import Any

import numpy as np


def surface_voxels(binary: np.ndarray) -> np.ndarray:
    """6-connected boundary voxels."""
    b = binary > 0.5
    surf = np.zeros_like(b, dtype=bool)
    for shift in (1, -1):
        for ax in range(3):
            n = np.roll(b, shift, axis=ax)
            surf |= b & ~n
    return surf


def _mean_min_dist_um(points_a: np.ndarray, points_b: np.ndarray, voxel_um: float) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return float("inf")
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2).min(axis=1)
    return float(dist.mean() * voxel_um)


def assd_um(binary_a: np.ndarray, binary_b: np.ndarray, *, voxel_um: float) -> float:
    """Average symmetric surface distance in micrometers."""
    sa = np.argwhere(surface_voxels(binary_a))
    sb = np.argwhere(surface_voxels(binary_b))
    d_ab = _mean_min_dist_um(sa, sb, voxel_um)
    d_ba = _mean_min_dist_um(sb, sa, voxel_um)
    return 0.5 * (d_ab + d_ba)


def thickness_map(binary: np.ndarray, axis: int = 2) -> np.ndarray:
    """2D thickness map: count of cured voxels along axis (Fig. 3)."""
    return binary.sum(axis=axis).astype(np.float64)


def msssim_stub(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Lightweight MS-SSIM proxy on 2D thickness maps."""
    a = map_a.astype(np.float64)
    b = map_b.astype(np.float64)
    if a.shape != b.shape:
        return 0.0
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    corr = float(np.mean(a * b))
    return float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))


def fidelity_metrics(
    binary: np.ndarray,
    target: np.ndarray,
    *,
    voxel_um: float,
) -> dict[str, float]:
    tm_pred = thickness_map(binary)
    tm_tgt = thickness_map(target)
    return {
        "assd_um": assd_um(binary, target, voxel_um=voxel_um),
        "ms_ssim": msssim_stub(tm_pred, tm_tgt),
    }


def table_uncorrected_fig3() -> dict[str, Any]:
    return {
        "notional_d_m2_s": 1.0e-10,
        "voxel_um": 47.0,
        "plate_feature_um": 500.0,
        "column_pillar_um": 600.0,
        "best_assd_near_d": 1.0e-10,
    }


def table_corrected_fig6() -> dict[str, Any]:
    return {
        "notional_d_m2_s": 1.0e-10,
        "peak_composite_d_m2_s": 3.4e-10,
        "improved_levels": [1.0e-10, 2.0e-10, 3.0e-10],
        "overcorrect_d_m2_s": 6.0e-10,
    }


def table_deconv_fig7() -> dict[str, Any]:
    return {
        "d_m2_s": 1.0e-10,
        "experimental_assd_um": 42.5,
        "experimental_ms_ssim": 0.696,
        "note": "RL deconv pre-comp inferior to co-optimization on uniformity",
    }


def operating_guidelines() -> list[str]:
    return [
        "Fit a single 3D kernel from uncorrected prints vs dose models (ASSD / MS-SSIM).",
        "Co-optimize diffusion with CT blur in BCLP gradient — avoid serial deconvolution.",
        "Use modified Radon/adjoint (Eq. 5–6) with FFT convolution for diffusion.",
        "Notional oxygen diffusivity D≈1×10⁻¹⁰ m²/s for PETA/TEMPO resin.",
        "Scale kernel time from dosing-rate ratios between corrected projections.",
        "Equal-volume thresholding for binarization; register micro-CT with shrinkage factor.",
    ]
