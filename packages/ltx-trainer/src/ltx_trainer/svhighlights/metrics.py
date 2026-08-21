"""SVHighlights evaluation metrics: mAP, HIT@1, HIT@K, IoU, window F1."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _binary_labels(labels: Sequence[int]) -> np.ndarray:
    return np.asarray(labels, dtype=np.int8).reshape(-1)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.sum() == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0
    precisions: list[float] = []
    for i, rel in enumerate(y_sorted, start=1):
        if rel:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def hit_at_k(scores: Sequence[float], labels: Sequence[int], k: int) -> float:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or k <= 0:
        return 0.0
    k = min(k, y.size)
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(int(y.sum()), 1))


def hit_at_one(scores: Sequence[float], labels: Sequence[int]) -> float:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.sum() == 0:
        return 0.0
    top = int(np.argmax(s))
    return float(y[top])


def temporal_iou(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    clip_duration_s: float = 2.0,
    threshold: float | None = None,
) -> float:
    """IoU between predicted highlight clips (top-K by score, K=#positives) and GT."""
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    k = n_pos
    top = set(np.argsort(-s)[:k].tolist())
    gt = set(np.where(y > 0)[0].tolist())
    if threshold is not None:
        top = set(np.where(s >= threshold)[0].tolist())
    union = top | gt
    if not union:
        return 0.0
    return float(len(top & gt) / len(union))


def window_f1(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    window_clips: int = 3,
) -> float:
    """Event-level F1 with ±window_clips tolerance (paper Appendix D)."""
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    k = n_pos
    pred_top = set(np.argsort(-s)[:k].tolist())

    def events(idxs: set[int]) -> list[tuple[int, int]]:
        if not idxs:
            return []
        sorted_i = sorted(idxs)
        ev: list[tuple[int, int]] = []
        start = prev = sorted_i[0]
        for i in sorted_i[1:]:
            if i == prev + 1:
                prev = i
            else:
                ev.append((start, prev))
                start = prev = i
        ev.append((start, prev))
        return ev

    gt_events = events(set(np.where(y > 0)[0].tolist()))
    pred_events = events(pred_top)
    tp = 0
    matched_gt: set[int] = set()
    for p_start, p_end in pred_events:
        for gi, (g_start, g_end) in enumerate(gt_events):
            if gi in matched_gt:
                continue
            if p_end + window_clips >= g_start and p_start - window_clips <= g_end:
                tp += 1
                matched_gt.add(gi)
                break
    fp = max(len(pred_events) - tp, 0)
    fn = max(len(gt_events) - tp, 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def evaluate_predictions(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    clip_duration_s: float = 2.0,
) -> dict[str, float]:
    y = _binary_labels(labels)
    n_pos = int(y.sum())
    return {
        "mAP": round(average_precision(scores, labels) * 100, 2),
        "HIT@1": round(hit_at_one(scores, labels) * 100, 2),
        "HIT@K": round(hit_at_k(scores, labels, n_pos) * 100, 2),
        "IoU": round(temporal_iou(scores, labels, clip_duration_s=clip_duration_s) * 100, 2),
        "window_F1": round(window_f1(scores, labels) * 100, 2),
    }
