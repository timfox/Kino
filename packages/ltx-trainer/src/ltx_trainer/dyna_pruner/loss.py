"""Unified objective (Eq. 1–4)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.dyna_pruner.masks import l1_sparsity_penalty


def mse_task_loss(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(diff * diff))


def total_loss(
    l_task: float,
    s: np.ndarray,
    importances: np.ndarray,
    *,
    lambda_d: float = 1e-3,
    lambda_w: float = 1e-3,
) -> dict[str, float]:
    """L_total = L_task + λd·||S||_1 + λw·||I||_1."""
    l_d = l1_sparsity_penalty(s)
    l_w = l1_sparsity_penalty(importances)
    return {
        "L_task": float(l_task),
        "L_data_sparsity": lambda_d * l_d,
        "L_model_sparsity": lambda_w * l_w,
        "L_total": float(l_task) + lambda_d * l_d + lambda_w * l_w,
    }
