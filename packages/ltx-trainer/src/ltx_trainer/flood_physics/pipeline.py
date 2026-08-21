"""Synthetic flood mapping demo — UNet+FNO+SWE stub."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.flood_physics.benchmarks import summary_anchors
from ltx_trainer.flood_physics.config import FloodPhysicsConfig
from ltx_trainer.flood_physics.losses import composite_loss, f1_score, focal_bce, iou_score, wet_mse
from ltx_trainer.flood_physics.model import hybrid_forward
from ltx_trainer.flood_physics.swe import swe_residuals


def _synthetic_tile(length: int = 16, seed: int = 0) -> tuple[list[float], list[float], list[float], list[int], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    sar = rng.normal(0.0, 0.3, length).tolist()
    optical = rng.normal(0.2, 0.25, length).tolist()
    dem = np.linspace(0.0, 0.5, length).tolist()
    gt_mask = [1 if i >= length // 3 and i < 2 * length // 3 else 0 for i in range(length)]
    gt_depth = [0.4 if m else 0.0 for m in gt_mask]
    gt_u = [0.1 if m else 0.0 for m in gt_mask]
    return sar, optical, dem, gt_mask, gt_depth, gt_u


def _grid_from_flat(flat: list[float], side: int) -> list[list[float]]:
    return [flat[i * side : (i + 1) * side] for i in range(side)]


def run_demo(cfg: FloodPhysicsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FloodPhysicsConfig()
    side = 8
    n = side * side
    sar, optical, dem, gt_mask, gt_depth, gt_u = _synthetic_tile(n)
    out = hybrid_forward(sar, optical, dem, cfg)
    pred_mask = [1 if p >= 0.5 else 0 for p in out["extent_p"]]
    wet = [m == 1 for m in gt_mask]

    l_ext = focal_bce(out["extent_p"], gt_mask, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    l_h = wet_mse(out["depth"], gt_depth, wet)
    l_u = wet_mse(out["u"], gt_u, wet)
    l_v = wet_mse(out["v"], [0.0] * n, wet)

    h_grid = _grid_from_flat(out["depth"], side)
    u_grid = _grid_from_flat(out["u"], side)
    v_grid = _grid_from_flat(out["v"], side)
    zb_grid = _grid_from_flat(dem, side)
    swe = swe_residuals(h_grid, u_grid, v_grid, zb_grid, n=cfg.manning_n, g=cfg.gravity)
    l_phys = swe["Rh_mse"] + swe["Ru_mse"] + swe["Rv_mse"]
    total = composite_loss(
        l_ext=l_ext,
        l_h=l_h,
        l_u=l_u,
        l_v=l_v,
        l_phys=l_phys,
        l_reg=0.0,
        cfg_weights={
            "ext": cfg.lambda_ext,
            "h": cfg.lambda_h,
            "u": cfg.lambda_u,
            "v": cfg.lambda_v,
            "phys": cfg.lambda_phys,
            "reg": cfg.lambda_reg,
        },
    )

    summary = summary_anchors()
    return {
        "config": {"lambda_phys": cfg.lambda_phys, "fno_modes": cfg.fno_modes},
        "loss": round(total, 6),
        "toy_iou": round(iou_score(pred_mask, gt_mask), 3),
        "toy_f1": round(f1_score(pred_mask, gt_mask), 3),
        "swe": {k: round(v, 6) if isinstance(v, float) else v for k, v in swe.items()},
        "paper_iou_anchor": summary["IoU"],
        "paper_f1_anchor": summary["F1"],
        "paper_depth_rmse_m": summary["depth_rmse_m"],
        "summary": summary,
    }
