"""CAMERA training objectives (Eq. 8–12)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def gating_entropy_loss(gates: Array, epsilon: float = 1e-8) -> float:
    """Eq. (8): encourage sharper expert usage (minimize entropy)."""
    g = np.clip(gates, epsilon, 1.0)
    ent = -np.mean(np.sum(g * np.log(g), axis=1))
    return float(ent)


def expert_loss(residuals_per_layer: list[dict[str, Array]]) -> float:
    """Eq. (11): mean squared norm of expert residuals across layers."""
    total = 0.0
    count = 0
    for layer_res in residuals_per_layer:
        for res in layer_res.values():
            total += float(np.mean(np.sum(res * res, axis=1)))
            count += 1
    return total / max(count, 1)


def oc_bce_loss(fraud_scores: Array) -> float:
    """Eq. (10): BCE(s_i, 0) pushing majority toward benign."""
    s = np.clip(fraud_scores, 1e-8, 1 - 1e-8)
    return float(-np.mean(np.log(1.0 - s)))


def total_loss(
    *,
    expert: float,
    gating: float,
    oc: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Eq. (12): L = L_expert + α·L_gating + β·L_OC."""
    return expert + alpha * gating + beta * oc
