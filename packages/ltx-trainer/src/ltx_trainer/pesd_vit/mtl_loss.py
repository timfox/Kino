"""Uncertainty-weighted multi-task objective (Eq. 2–3)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def homoscedastic_mtl_loss(
    task_losses: Sequence[float],
    log_sigma2: Sequence[float],
) -> float:
    """
    L_MTL = sum_t (1/(2σ²_t) L_t + 1/2 log σ²_t).

    log_sigma2 are learnable log-variance parameters (clamped externally).
    """
    total = 0.0
    for lt, ls2 in zip(task_losses, log_sigma2, strict=True):
        sigma2 = math.exp(float(ls2))
        total += 0.5 * lt / max(sigma2, 1e-8) + 0.5 * float(ls2)
    return total


def total_training_loss(
    task_losses: Sequence[float],
    log_sigma2: Sequence[float],
    ortho_loss: float,
    *,
    ortho_lambda: float,
    manual_weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """L_total = L_MTL + λ L_ortho with optional manual task scaling on L_t."""
    scaled = list(task_losses)
    if manual_weights is not None:
        scaled = [lt * w for lt, w in zip(task_losses, manual_weights, strict=True)]
    l_mtl = homoscedastic_mtl_loss(scaled, log_sigma2)
    l_total = l_mtl + ortho_lambda * ortho_loss
    return {
        "l_mtl": float(l_mtl),
        "l_ortho": float(ortho_loss),
        "l_total": float(l_total),
        "ortho_lambda": float(ortho_lambda),
    }


def clamp_log_sigma2(
    log_sigma2: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> np.ndarray:
    return np.clip(log_sigma2, lo, hi)
