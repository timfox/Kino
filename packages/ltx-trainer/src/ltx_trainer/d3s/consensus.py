"""D3S Consensus: coupled D3 + stereo depth selection (Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.d3s.calibration import shift_for_virtual_baseline, shift_image
from ltx_trainer.d3s.config import D3SConfig, D3SConsensusParams, D3SOptics
from ltx_trainer.d3s.d3 import d3_depth_eq17


@dataclass
class D3SResult:
    depth_m: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    candidate_depths_m: np.ndarray


def candidate_depths(params: D3SConsensusParams) -> np.ndarray:
    n = int(np.round((params.z_max_m - params.z_min_m) / params.z_step_m)) + 1
    return np.linspace(params.z_min_m, params.z_max_m, max(n, 2))


def confidence_for_candidate(
    z_i: float,
    z_d3_values: np.ndarray,
    *,
    delta_z: float,
    delta_inv_z: float,
) -> tuple[float, bool]:
    """Eq. 13–14: consensus on Z and 1/Z."""
    dz = z_i - z_d3_values
    dr = (1.0 / max(z_i, 1e-6)) - (1.0 / np.maximum(z_d3_values, 1e-6))
    inf_z = float(np.max(np.abs(dz))) if dz.size else float("inf")
    inf_r = float(np.max(np.abs(dr))) if dr.size else float("inf")
    c = (1.0 + inf_z) * (1.0 + inf_r)
    c = 1.0 / c if c > 0 else 0.0
    ok = inf_z < delta_z and inf_r < delta_inv_z
    return c, ok


def d3s_consensus(
    i0: np.ndarray,
    i1: np.ndarray,
    *,
    cfg: D3SConfig | None = None,
) -> D3SResult:
    """Sparse depth map from dual-defocus stereo pair."""
    cfg = cfg or D3SConfig()
    optics = cfg.optics
    params = cfg.consensus
    z_cands = candidate_depths(params)
    h, w = i0.shape
    best_z = np.full((h, w), np.nan, dtype=np.float64)
    best_c = np.zeros((h, w), dtype=np.float64)

    for z_i in z_cands:
        z_d3_stack: list[np.ndarray] = []
        for vbx in optics.virtual_baselines_mm:
            dx = shift_for_virtual_baseline(z_i, vbx, optics)
            i1s = shift_image(i1, dx)
            z_d3 = d3_depth_eq17(i0, i1s, delta_x_mm=vbx, window_radius=params.window_radius)
            z_d3_stack.append(z_d3)
        z_d3_mean = np.nanmean(np.stack(z_d3_stack, axis=0), axis=0)
        # Per-pixel confidence from multi-baseline spread.
        dz = np.abs(z_i - z_d3_mean)
        dr = np.abs((1.0 / max(z_i, 1e-6)) - (1.0 / np.maximum(z_d3_mean, 1e-6)))
        c_map = 1.0 / ((1.0 + dz) * (1.0 + dr))
        improve = c_map > best_c
        best_c = np.where(improve, c_map, best_c)
        best_z = np.where(improve, z_i, best_z)

    valid = best_c >= params.confidence_threshold
    depth = np.where(valid, best_z, np.nan)
    return D3SResult(
        depth_m=depth,
        confidence=best_c,
        valid_mask=valid,
        candidate_depths_m=z_cands,
    )


def mean_absolute_error_cm(pred: np.ndarray, gt_m: float) -> float:
    mask = np.isfinite(pred)
    if not np.any(mask):
        return float("inf")
    err_m = np.abs(pred[mask] - gt_m)
    return float(np.mean(err_m) * 100.0)
