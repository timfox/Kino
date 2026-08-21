"""Linear probing scores and baseline normalization (Sec. 3.3)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.sarl.tasks import ProbeTask


def baseline_normalize(score: float, baseline: float) -> float:
    """φ(x; b) = (x − b) / (1 − b)."""
    if abs(1.0 - baseline) < 1e-8:
        return 0.0
    return (score - baseline) / (1.0 - baseline)


def continuous_score(mae: float, value_range: tuple[float, float]) -> float:
    """Score = 1 − MAE/R for continuous factors."""
    lo, hi = value_range
    span = hi - lo
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - mae / span))


def random_baseline(task: ProbeTask) -> float:
    if task.kind == "categorical":
        return 1.0 / task.bins
    return 0.0


def aggregate_group(scores: dict[str, float], tasks: tuple[ProbeTask, ...]) -> float:
    normalized = [
        baseline_normalize(scores[t.name], random_baseline(t))
        for t in tasks
        if t.name in scores
    ]
    if not normalized:
        return 0.0
    return float(np.mean(normalized))


def linear_probe_predict(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> float:
    """Tiny ridge classifier macro-F1 stub on frozen embeddings."""
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape
    w = rng.standard_normal((d, int(labels.max()) + 1))
    logits = embeddings @ w
    preds = logits.argmax(axis=1)
    classes = np.unique(labels)
    f1s = []
    for c in classes:
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))
