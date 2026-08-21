"""Unified GRPO + segmentation objectives (Sec. 3.3, Eq. 6)."""

from __future__ import annotations

import math


def dice_coefficient(pred: float, target: float, *, eps: float = 1e-6) -> float:
    """Scalar Dice for smoke tests."""
    return (2.0 * pred * target + eps) / (pred + pred + target + target + eps)


def dice_loss(pred: float, target: float) -> float:
    return 1.0 - dice_coefficient(pred, target)


def bce_scalar(pred: float, target: float) -> float:
    p = min(max(pred, 1e-6), 1.0 - 1e-6)
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def segmentation_loss(
    heatmap_match: float,
    mask_iou: float,
    *,
    lambda_dice: float = 0.6,
) -> float:
    """LSEG = LBCE(heatmap) + λD·LDICE(mask) — scalar toy."""
    return bce_scalar(heatmap_match, 1.0) + lambda_dice * dice_loss(mask_iou, mask_iou)


def confidence_loss(confidence: float, matched: bool) -> float:
    """Per-slot BCE on confidence vs. bipartite match indicator."""
    y = 1.0 if matched else 0.0
    return bce_scalar(confidence, y)


def total_objective(
    grpo: float,
    seg: float,
    conf: float,
    *,
    lambda_s: float = 1.0,
    lambda_c: float = 0.2,
) -> float:
    r"""L = LGRPO + λS·LSEG + λC·LCONF (Eq. 6)."""
    return grpo + lambda_s * seg + lambda_c * conf
