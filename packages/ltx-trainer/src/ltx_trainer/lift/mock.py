"""LiFT smoke evaluation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lift.config import LiftConfig
from ltx_trainer.lift.features import drift_target_mean, fourier_depth_encoding


def evaluation_smoke() -> dict[str, Any]:
    cfg = LiftConfig()
    enc = fourier_depth_encoding(5)
    drift = drift_target_mean([[1.0, 0.2]], [[1.1, 0.15]])
    return {
        "depth_encoding_dim": len(enc),
        "drift_l2": round(float(drift["drift_l2"]), 4),
        "best_unconditional_method": "LiFT-U",
        "lift_u_params_m": cfg.lift_u_mapper_params_m,
    }
