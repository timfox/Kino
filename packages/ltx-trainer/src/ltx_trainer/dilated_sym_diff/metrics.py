"""IoU and radius tradeoff analysis — Sec. 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dilated_sym_diff.morphology import dilated_symmetric_difference


def iou(x: np.ndarray, y: np.ndarray) -> float:
    """IoU(X, Y) = |X ∩ Y| / |X ∪ Y|."""
    xx = x.astype(bool)
    yy = y.astype(bool)
    inter = int(np.logical_and(xx, yy).sum())
    union = int(np.logical_or(xx, yy).sum())
    if union == 0:
        return 1.0
    return inter / union


def iou_vs_radius(
    a: np.ndarray,
    b: np.ndarray,
    reference: np.ndarray,
    radii: list[int] | range,
) -> list[tuple[int, float]]:
    """IoU between dilated symmetric difference and reference mask vs r."""
    return [(r, iou(dilated_symmetric_difference(a, b, r), reference)) for r in radii]


def choose_radius(
    curve: list[tuple[int, float]],
    *,
    delta_align: float,
    target_iou: float = 0.95,
) -> int | None:
    """Pick smallest r > δ_align with IoU ≥ target."""
    candidates = [r for r, score in curve if r > delta_align and score >= target_iou]
    return min(candidates) if candidates else None


def metrics_card() -> dict[str, Any]:
    return {
        "iou": "|X ∩ Y| / |X ∪ Y|",
        "analysis": "IoU vs dilation radius r for tradeoff (Sec. 3)",
        "lower_bound": "r > δ_align compensates registration error",
        "upper_bound": "large r engulfs nearby pattern regions",
    }
