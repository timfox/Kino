"""CapBench event recall/precision metrics (arXiv:2606.05748)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


class EntailmentRelation(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


def event_recall(
    ground_truth_events: Sequence[str],
    predicted_events: Sequence[str],
    *,
    entailed: Sequence[bool] | None = None,
) -> float:
    """Fraction of GT events entailed by predicted caption events."""
    gt = list(ground_truth_events)
    if not gt:
        return 100.0
    if entailed is not None and len(entailed) == len(gt):
        hit = sum(1 for e in entailed if e)
        return round(100.0 * hit / len(gt), 1)
    pred_text = " ".join(predicted_events).lower()
    hit = sum(1 for e in gt if _token_overlap(e, pred_text) >= 0.35)
    return round(100.0 * hit / len(gt), 1)


def event_precision(
    ground_truth_events: Sequence[str],
    predicted_events: Sequence[str],
    *,
    entailed: Sequence[bool] | None = None,
) -> float:
    """Fraction of predicted events supported by ground truth."""
    pred = list(predicted_events)
    if not pred:
        return 0.0
    if entailed is not None and len(entailed) == len(pred):
        hit = sum(1 for e in entailed if e)
        return round(100.0 * hit / len(pred), 1)
    gt_text = " ".join(ground_truth_events).lower()
    hit = sum(1 for e in pred if _token_overlap(e, gt_text) >= 0.35)
    return round(100.0 * hit / len(pred), 1)


def violative_recall(
    violative_gt: Sequence[str],
    predicted_events: Sequence[str],
    *,
    entailed: Sequence[bool] | None = None,
) -> float:
    return event_recall(violative_gt, predicted_events, entailed=entailed)


def non_violative_recall(
    non_violative_gt: Sequence[str],
    predicted_events: Sequence[str],
    *,
    entailed: Sequence[bool] | None = None,
) -> float:
    return event_recall(non_violative_gt, predicted_events, entailed=entailed)


def capbench_f1(recall_pct: float, precision_pct: float) -> float:
    r, p = recall_pct / 100.0, precision_pct / 100.0
    if r + p == 0:
        return 0.0
    return round(200.0 * r * p / (r + p), 1)


def capbench_bundle(
    ground_truth_events: Sequence[str],
    predicted_events: Sequence[str],
    *,
    violative_gt: Sequence[str] | None = None,
    non_violative_gt: Sequence[str] | None = None,
) -> dict[str, float]:
    violative_gt = list(violative_gt or [])
    non_violative_gt = list(non_violative_gt or [])
    rec = event_recall(ground_truth_events, predicted_events)
    prec = event_precision(ground_truth_events, predicted_events)
    out: dict[str, float] = {
        "recall": rec,
        "precision": prec,
        "f1": capbench_f1(rec, prec),
    }
    if violative_gt:
        out["vio_rec"] = violative_recall(violative_gt, predicted_events)
    if non_violative_gt:
        out["non_vio_rec"] = non_violative_recall(non_violative_gt, predicted_events)
    return out


def _token_overlap(event: str, corpus: str) -> float:
    tokens = {t for t in event.lower().split() if len(t) > 3}
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in corpus) / len(tokens)
