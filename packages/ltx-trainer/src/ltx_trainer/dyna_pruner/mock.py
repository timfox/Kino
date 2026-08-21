"""Toy spatio-temporal tensors and evaluation smoke."""

from __future__ import annotations

from typing import Any

import numpy as np


def toy_weatherbench_frame(*, t: int = 8, h: int = 32, w: int = 32) -> np.ndarray:
    """
    Synthetic radar-like sequence: calm background + localized storm cell.

    Shape ``(T, C, H, W)`` with C=1.
    """
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 0.02, size=(t, 1, h, w))
    cy, cx = h // 3, w // 2
    for ti in range(t):
        shift = ti * 0.8
        yy, xx = np.ogrid[:h, :w]
        dist = (yy - (cy + shift)) ** 2 + (xx - cx) ** 2
        storm = np.exp(-dist / (2 * 4.0**2)) * (0.6 + 0.1 * ti)
        base[ti, 0] += storm
    return base.astype(np.float64)


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.dyna_pruner.config import DynaPrunerConfig
    from ltx_trainer.dyna_pruner.loss import mse_task_loss, total_loss
    from ltx_trainer.dyna_pruner.masks import apply_data_mask, importance_from_temporal_variance, mask_summary
    from ltx_trainer.dyna_pruner.synergy import synchronized_masks

    cfg = DynaPrunerConfig()
    x = toy_weatherbench_frame()
    s = importance_from_temporal_variance(x)
    sync = synchronized_masks(
        s,
        sd=cfg.data_sparsity_sd,
        sw=cfg.model_sparsity_sw,
        ste_threshold=cfg.ste_threshold,
        kernel=cfg.receptive_field,
    )
    x_sparse = apply_data_mask(x, sync["M_data"])
    # Mock one-step prediction: sparse input ≈ damped target
    target = x[-1]
    pred = x_sparse[-1]
    l_task = mse_task_loss(pred, target)
    losses = total_loss(
        l_task,
        sync["S"],
        sync["I"],
        lambda_d=cfg.lambda_d,
        lambda_w=cfg.lambda_w,
    )
    summary = mask_summary(s, threshold=sync["data_threshold"])
    return {
        "data_sparsity": summary["data_sparsity"],
        "model_keep_ratio": float(np.mean(sync["M_weight"])),
        "storm_pixels_active": summary["active_pixels"],
        "L_total": round(losses["L_total"], 6),
        "tau": round(float(sync["tau"]), 4),
    }
