"""CAFNet training losses (Eq. 1)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig, DeepfakeClass


def weighted_cross_entropy(
    logits: np.ndarray,
    target: int,
    *,
    weights: tuple[float, float, float] | None = None,
) -> float:
    cfg = CafNetConfig()
    w = weights or cfg.class_weights
    logits = np.asarray(logits, dtype=np.float64)
    log_probs = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
    probs = np.exp(log_probs)
    return float(-w[target] * np.log(probs[target] + 1e-12))


def boundary_mse(
    pred: tuple[float, float],
    true: tuple[float, float],
) -> float:
    p = np.array(pred, dtype=np.float64)
    t = np.array(true, dtype=np.float64)
    return float(np.mean((p - t) ** 2))


def cafnet_loss(
    logits_main: np.ndarray,
    logits_aux: np.ndarray,
    target_class: int,
    *,
    boundary_pred: tuple[float, float] | None = None,
    boundary_true: tuple[float, float] | None = None,
    cfg: CafNetConfig | None = None,
) -> dict[str, float]:
    """L = Lcls + 0.4 Laux + 0.3 Ltemp (Ltemp only for half-truth)."""
    cfg = cfg or CafNetConfig()
    l_cls = weighted_cross_entropy(logits_main, target_class)
    l_aux = weighted_cross_entropy(logits_aux, target_class)
    l_temp = 0.0
    if target_class == 2 and boundary_pred is not None and boundary_true is not None:
        l_temp = boundary_mse(boundary_pred, boundary_true)
    total = l_cls + cfg.aux_loss_weight * l_aux + cfg.temp_loss_weight * l_temp
    return {
        "Lcls": round(l_cls, 4),
        "Laux": round(l_aux, 4),
        "Ltemp": round(l_temp, 4),
        "total": round(total, 4),
    }


def class_index(label: DeepfakeClass) -> int:
    return {
        DeepfakeClass.REAL: 0,
        DeepfakeClass.FAKE: 1,
        DeepfakeClass.HALF_TRUTH: 2,
    }[label]
