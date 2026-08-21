"""Evaluation metrics and confusion-matrix helpers (§2.7.3)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def accuracy(preds: list[str], gts: list[str], *, unknown_as_wrong: bool = True) -> float:
    """Overall accuracy; unknown counts as wrong when unknown_as_wrong=True (§2.6)."""
    if not preds or len(preds) != len(gts):
        return 0.0
    correct = 0
    for p, g in zip(preds, gts, strict=True):
        if p == g:
            correct += 1
        elif not unknown_as_wrong and p == "unknown":
            continue
    return correct / len(gts)


def per_class_prf(
    preds: list[str],
    gts: list[str],
    classes: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1."""
    classes = list(classes)
    out: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts, strict=True) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts, strict=True) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts, strict=True) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1}
    return out


def confusion_counts(
    preds: list[str],
    gts: list[str],
    classes: list[str],
) -> dict[str, dict[str, int]]:
    """Confusion matrix as nested count dict [gt][pred]."""
    matrix: dict[str, dict[str, int]] = {g: defaultdict(int) for g in classes}
    for p, g in zip(preds, gts, strict=True):
        if g in matrix:
            matrix[g][p] += 1
    return {g: dict(row) for g, row in matrix.items()}
