"""Classification metrics — §3.1."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return 100.0 * 0.5 * (sens + spec)


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return 100.0 * tp / max(tp + fn, 1), 100.0 * tn / max(tn + fp, 1)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def auroc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Toy AUROC via Mann–Whitney U on binary labels."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
