"""Temporal detection metrics for attribution maps (arXiv:2605.23293)."""

from __future__ import annotations

import numpy as np


def temporal_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU = |M_attr ∩ M_GT| / |M_attr ∪ M_GT|."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("pred_mask and gt_mask must have the same shape")
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if union == 0.0:
        return 1.0 if inter == 0.0 else 0.0
    return inter / union


def frame_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    """Frame-level precision, recall, F1."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def pointing_game(frame_attr: np.ndarray, gt_mask: np.ndarray) -> bool:
    """True if argmax |attr| falls inside a ground-truth active frame."""
    frame_attr = np.asarray(frame_attr, dtype=np.float64)
    gt = np.asarray(gt_mask, dtype=bool)
    if frame_attr.size == 0:
        return False
    t_max = int(np.argmax(np.abs(frame_attr)))
    return bool(gt[t_max]) if t_max < gt.size else False


def sweep_percentile_metrics(
    frame_attr: np.ndarray,
    gt_mask: np.ndarray,
    percentiles: range | None = None,
) -> list[dict[str, float]]:
    """IoU and F1 across percentile thresholds (validation sweep)."""
    percentiles = percentiles or range(1, 100)
    rows: list[dict[str, float]] = []
    for pct in percentiles:
        from ltx_trainer.ig_sed.ig import binarize_percentile

        mask = binarize_percentile(frame_attr, float(pct))
        _, _, f1 = frame_f1(mask, gt_mask)
        rows.append({"percentile": float(pct), "iou": temporal_iou(mask, gt_mask), "f1": f1})
    return rows


def best_percentile(rows: list[dict[str, float]], metric: str = "iou") -> dict[str, float]:
    """Return row with maximum metric value."""
    if not rows:
        raise ValueError("rows must be non-empty")
    key = metric if metric in ("iou", "f1") else "iou"
    return max(rows, key=lambda r: r[key])
