"""Synthetic dual-defocus stereo pairs and evaluation sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.d3s.calibration import shift_for_virtual_baseline, shift_image
from ltx_trainer.d3s.config import D3SConfig, D3SConsensusParams
from ltx_trainer.d3s.consensus import d3s_consensus, mean_absolute_error_cm


def _gaussian_kernel(sigma: float, *, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    sigma = min(float(sigma), 8.0)
    r = radius if radius is not None else max(1, min(12, int(3 * sigma + 0.5)))
    xs = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    return k


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel(sigma)
    tmp = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, img)
    return np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, tmp)


def defocus_sigma(depth_m: float, *, optics: D3SConfig) -> tuple[float, float]:
    """Depth-dependent blur in pixels (stub — small differential defocus)."""
    _ = optics
    # Mild depth-varying blur; views differ slightly (dual-focus prototype).
    base = 1.2 + 0.35 * abs(depth_m - 0.85)
    return max(0.6, base * 0.92), max(0.6, base * 1.08)


def synthetic_stereo_pair(
    depth_m: float,
    *,
    height: int = 64,
    width: int = 96,
    seed: int = 0,
    cfg: D3SConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Front-parallel textured plane with stereo shift + differential defocus."""
    cfg = cfg or D3SConfig()
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    texture = 0.55 + 0.35 * np.sin(0.25 * x + 0.17 * y) + 0.05 * rng.normal(size=(height, width))
    sig0, sig1 = defocus_sigma(depth_m, optics=cfg)
    i0 = gaussian_blur(texture, sig0)
    disparity = shift_for_virtual_baseline(depth_m, 0.0, cfg.optics)
    i1_sharp = shift_image(texture, disparity)
    i1 = gaussian_blur(i1_sharp, sig1)
    return i0.astype(np.float64), i1.astype(np.float64)


def evaluate_depth_sweep(
    depths_m: np.ndarray,
    *,
    cfg: D3SConfig | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """MAE vs depth for synthetic planes (Fig. 5a-style stub)."""
    cfg = cfg or D3SConfig()
    rows: list[dict[str, Any]] = []
    for i, z in enumerate(depths_m):
        i0, i1 = synthetic_stereo_pair(z, seed=seed + i, cfg=cfg)
        result = d3s_consensus(i0, i1, cfg=cfg)
        mae = mean_absolute_error_cm(result.depth_m, z)
        density = float(np.mean(result.valid_mask))
        rows.append({"depth_m": float(z), "mae_cm": mae, "density": density})
    return rows


def working_range_from_sweep(rows: list[dict[str, Any]], *, mae_limit_cm: float = 1.0) -> tuple[float, float] | None:
    good = [r for r in rows if r["mae_cm"] < mae_limit_cm and r["density"] > 0.01]
    if not good:
        return None
    zs = [r["depth_m"] for r in good]
    return float(min(zs)), float(max(zs))


def full_pipeline_demo(*, seed: int = 3, cfg: D3SConfig | None = None) -> dict[str, Any]:
    cfg = cfg or D3SConfig()
    z_test = 0.8
    i0, i1 = synthetic_stereo_pair(z_test, seed=seed, cfg=cfg)
    result = d3s_consensus(i0, i1, cfg=cfg)
    depths = np.linspace(0.35, 1.5, 12)
    sweep = evaluate_depth_sweep(depths, cfg=cfg, seed=seed + 10)
    wr = working_range_from_sweep(sweep, mae_limit_cm=cfg.mae_target_cm)
    return {
        "test_depth_m": z_test,
        "mae_cm": mean_absolute_error_cm(result.depth_m, z_test),
        "valid_fraction": float(np.mean(result.valid_mask)),
        "max_confidence": float(np.nanmax(result.confidence)),
        "sweep": sweep,
        "working_range_m": wr,
    }


def confidence_threshold_ablation(
    *,
    cfg: D3SConfig | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
) -> list[dict[str, Any]]:
    """Fig. 5b-style density vs threshold on a fixed scene."""
    base = cfg or D3SConfig()
    i0, i1 = synthetic_stereo_pair(0.9, seed=11, cfg=base)
    rows: list[dict[str, Any]] = []
    for th in thresholds:
        params = D3SConsensusParams(confidence_threshold=th)
        tuned = D3SConfig(optics=base.optics, consensus=params)
        out = d3s_consensus(i0, i1, cfg=tuned)
        rows.append(
            {
                "confidence_threshold": th,
                "density": float(np.mean(out.valid_mask)),
                "mae_cm": mean_absolute_error_cm(out.depth_m, 0.9),
            }
        )
    return rows
