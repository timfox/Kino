"""Toy evaluation metrics for SpecX-style tasks."""

from __future__ import annotations

import numpy as np


def top_k_accuracy(
    candidates: list[list[str]],
    ground_truth: list[str],
    *,
    k: int = 1,
) -> float:
    """Fraction of samples where truth appears in top-k candidate lists."""
    if not candidates:
        return 0.0
    hits = 0
    for cands, truth in zip(candidates, ground_truth, strict=True):
        if truth in cands[:k]:
            hits += 1
    return hits / len(candidates)


def macro_f1_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Macro-averaged F1 for binary multilabel (n_samples, n_classes)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    f1s: list[float] = []
    for c in range(y_true.shape[1]):
        tp = np.sum(y_true[:, c] & y_pred[:, c])
        fp = np.sum(~y_true[:, c] & y_pred[:, c])
        fn = np.sum(y_true[:, c] & ~y_pred[:, c])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)
