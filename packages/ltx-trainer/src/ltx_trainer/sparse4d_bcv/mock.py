"""Smoke helpers for sparse 4D BCV."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sparse4d_bcv.bootstrap import bootstrap_cv_summary, mse, normalized_cross_correlation
from ltx_trainer.sparse4d_bcv.config import Sparse4dBcvConfig
from ltx_trainer.sparse4d_bcv.nyquist import (
    evenly_spaced_projection_angles,
    interlaced_time_indices,
    nyquist_velocity,
)


def evaluation_smoke() -> dict[str, Any]:
    cfg = Sparse4dBcvConfig()
    rng = np.random.default_rng(2)
    t, h, w = 32, 16, 16
    gt = rng.standard_normal((t, h, w))
    stats = bootstrap_cv_summary(gt, num_subsets=20, noise=0.1, seed=2)
    recon = gt + 0.05 * rng.standard_normal(gt.shape)
    return {
        "dataset": cfg.dataset_name,
        "mse_toy": round(mse(recon, gt), 4),
        "ncc_toy": round(normalized_cross_correlation(recon, gt), 4),
        **{k: round(v, 4) for k, v in stats.items()},
        "interlaced_even": len(interlaced_time_indices(t)[0]),
    }


def nyquist_demo(*, delta_x_um: float = 1.0, delta_t_ms: float = 1.0) -> dict[str, Any]:
    v = nyquist_velocity(delta_x=delta_x_um, delta_t=delta_t_ms)
    return {"delta_x": delta_x_um, "delta_t_ms": delta_t_ms, "v_nyquist_per_ms": v}


def bootstrap_demo() -> dict[str, Any]:
    cfg = Sparse4dBcvConfig()
    angles = evenly_spaced_projection_angles(4)
    return {"dataset": cfg.dataset_name, "ultrasparse_angles": angles}
