"""Multi-objective flood loss — Eqs. 5–7."""

from __future__ import annotations

import math
from typing import Any


def focal_bce(pred: list[float], target: list[int], *, alpha: float = 0.25, gamma: float = 2.0) -> float:
    """Lext — Eq. 6 (per-pixel stub)."""
    if not pred:
        return 0.0
    total = 0.0
    for p_raw, y in zip(pred, target):
        p = min(max(p_raw, 1e-6), 1.0 - 1e-6)
        if y == 1:
            total += -alpha * ((1 - p) ** gamma) * math.log(p)
        else:
            total += -(1 - alpha) * (p ** gamma) * math.log(1 - p)
    return total / len(pred)


def wet_mse(pred: list[float], target: list[float], wet_mask: list[bool]) -> float:
    """Lh/Lu/Lv on wet cells — Eq. 7."""
    errs = [(p - t) ** 2 for p, t, w in zip(pred, target, wet_mask) if w]
    return sum(errs) / max(len(errs), 1)


def composite_loss(
    *,
    l_ext: float,
    l_h: float,
    l_u: float,
    l_v: float,
    l_phys: float,
    l_reg: float,
    cfg_weights: dict[str, float] | None = None,
) -> float:
    """L — Eq. 5."""
    w = cfg_weights or {
        "ext": 1.0,
        "h": 1.0,
        "u": 0.5,
        "v": 0.5,
        "phys": 0.1,
        "reg": 1e-4,
    }
    return (
        w["ext"] * l_ext
        + w["h"] * l_h
        + w["u"] * l_u
        + w["v"] * l_v
        + w["phys"] * l_phys
        + w["reg"] * l_reg
    )


def iou_score(pred_mask: list[int], gt_mask: list[int]) -> float:
    inter = sum(1 for p, g in zip(pred_mask, gt_mask) if p and g)
    union = sum(1 for p, g in zip(pred_mask, gt_mask) if p or g)
    return inter / max(union, 1)


def f1_score(pred_mask: list[int], gt_mask: list[int]) -> float:
    tp = sum(1 for p, g in zip(pred_mask, gt_mask) if p and g)
    fp = sum(1 for p, g in zip(pred_mask, gt_mask) if p and not g)
    fn = sum(1 for p, g in zip(pred_mask, gt_mask) if not p and g)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)


def loss_components_card() -> dict[str, Any]:
    return {
        "Lext": "focal binary cross-entropy on flood extent (Eq. 6)",
        "Lh_Lu_Lv": "wet-cell MSE for depth and velocity (Eq. 7)",
        "Lphys": "SWE residual penalties Rh, Ru, Rv (Eq. 4)",
        "Lreg": "weight decay",
        "composite": "λ-weighted sum (Eq. 5)",
    }
