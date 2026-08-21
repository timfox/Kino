"""Table II and hydrodynamic RMSE anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.flood_physics.constants import (
    HYDRO_METRICS,
    HYBRID_ROW,
    PHYSICS_ABLATION_DEPTH_RMSE_INCREASE_PCT,
    TABLE2_EXTENT,
)


def table_2_extent() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_EXTENT]


def hydro_metrics_card() -> dict[str, Any]:
    return dict(HYDRO_METRICS)


def summary_anchors() -> dict[str, Any]:
    hybrid = dict(HYBRID_ROW)
    unet = next(r for r in TABLE2_EXTENT if r["model"] == "UNet-only")
    fno = next(r for r in TABLE2_EXTENT if r["model"] == "FNO-only")
    return {
        "IoU": float(hybrid["IoU"]),
        "F1": float(hybrid["F1"]),
        "IoU_gain_vs_unet": round(float(hybrid["IoU"]) - float(unet["IoU"]), 2),
        "IoU_gain_vs_fno": round(float(hybrid["IoU"]) - float(fno["IoU"]), 2),
        "depth_rmse_m": HYDRO_METRICS["depth_rmse_m"],
        "velocity_rmse_m_s": HYDRO_METRICS["velocity_rmse_m_s"],
        "mass_imbalance_pct": HYDRO_METRICS["mass_imbalance_pct"],
        "physics_ablation_depth_rmse_increase_pct": PHYSICS_ABLATION_DEPTH_RMSE_INCREASE_PCT,
    }
