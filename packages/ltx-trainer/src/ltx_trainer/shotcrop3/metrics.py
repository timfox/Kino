"""TSC-Bench metrics: IoU, BDE, Unipercent proxies, overall score."""

from __future__ import annotations

import numpy as np

from ltx_trainer.shotcrop3.geometry import boundary_displacement_error, iou


def evaluate_triple(
    preds: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
) -> dict[str, float]:
    ious = [iou(preds[k], gts[k]) for k in gts]
    bdes = [boundary_displacement_error(preds[k], gts[k]) for k in gts]
    return {
        "iou": float(np.mean(ious)),
        "bde": float(np.mean(bdes)),
    }


def unipercent_proxy(iou_mean: float, *, aesthetic: float) -> dict[str, float]:
    """Table 1 IAA/IQA/ISTA proxies from alignment + aesthetic."""
    iaa = float(np.clip(0.35 + 0.35 * aesthetic + 0.15 * iou_mean, 0.0, 1.0))
    iqa = float(np.clip(0.38 + 0.32 * aesthetic + 0.12 * iou_mean, 0.0, 1.0))
    ista = float(np.clip(0.28 + 0.40 * iou_mean + 0.10 * aesthetic, 0.0, 1.0))
    return {"iaa": iaa, "iqa": iqa, "ista": ista}


def overall_score(
    *,
    iou_mean: float,
    aesthetic: float,
    storytelling: float,
) -> dict[str, float]:
    uni = unipercent_proxy(iou_mean, aesthetic=aesthetic)
    overall = float(np.clip(0.45 * aesthetic + 0.35 * storytelling + 0.20 * iou_mean, 0.0, 1.0))
    return {
        **uni,
        "aesthetic": aesthetic,
        "storytelling": storytelling,
        "overall": overall,
    }
