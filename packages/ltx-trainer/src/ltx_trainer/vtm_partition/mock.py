"""VTM partition RL smoke helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vtm_partition.features import CUContext
from ltx_trainer.vtm_partition.qtmtt import ALL_SPLITS


def sample_cu_context(seed: int = 0) -> CUContext:
    _ = seed
    return CUContext(
        width=64,
        height=64,
        qp=32,
        ns_rd_cost=100.0,
        ns_rate=0.5,
        ns_distortion=40.0,
        parent_ns_rd_cost=120.0,
        top_rd_cost=90.0,
        left_rd_cost=85.0,
        top_qt_depth=2,
        left_qt_depth=1,
    )


def sample_ground_truth_costs(seed: int = 0) -> tuple[dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(seed)
    l1 = {s: 1000.0 + float(rng.standard_normal()) for s in ALL_SPLITS}
    l2 = {s: 500.0 + float(rng.standard_normal()) for s in ALL_SPLITS}
    return l1, l2


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.vtm_partition.rl_agent import mse_level1, predict_q_values_linear, select_splits_top_n

    ctx = sample_cu_context(seed)
    l1, _l2 = sample_ground_truth_costs(seed)
    state = np.array([ctx.qp, ctx.ns_rd_cost, ctx.top_rd_cost], dtype=np.float64)
    pred = predict_q_values_linear(state, seed=seed)
    picks = select_splits_top_n(pred, top_n=3)
    return {"qp": ctx.qp, "top_splits": picks, "mse_l1": round(mse_level1(pred, l1), 2)}
