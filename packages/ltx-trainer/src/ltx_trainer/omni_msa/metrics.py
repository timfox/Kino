"""MSA regression/classification and reliability metrics (arXiv:2606.05713)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def pearson_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def acc7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Seven-class accuracy on rounded sentiment bins."""
    bins = np.round(np.clip(np.asarray(y_true), -3, 3)).astype(int)
    pred_bins = np.round(np.clip(np.asarray(y_pred), -3, 3)).astype(int)
    return float(np.mean(bins == pred_bins) * 100.0) if bins.size else 0.0


def acc2_nonzero(y_true: Sequence[float], y_pred: Sequence[float], *, eps: float = 0.1) -> float:
    """Binary accuracy on non-neutral subset (negative vs positive)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(yt) > eps
    if not np.any(mask):
        return 0.0
    true_bin = yt[mask] > 0
    pred_bin = yp[mask] > 0
    return float(np.mean(true_bin == pred_bin) * 100.0)


def f1_nonzero(y_true: Sequence[float], y_pred: Sequence[float], *, eps: float = 0.1) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(yt) > eps
    if not np.any(mask):
        return 0.0
    true_bin = (yt[mask] > 0).astype(int)
    pred_bin = (yp[mask] > 0).astype(int)
    tp = int(np.sum((true_bin == 1) & (pred_bin == 1)))
    fp = int(np.sum((true_bin == 0) & (pred_bin == 1)))
    fn = int(np.sum((true_bin == 1) & (pred_bin == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec) * 100.0)


def regression_bundle(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "mae": round(mae(y_true, y_pred), 3),
        "corr": round(pearson_corr(y_true, y_pred), 3),
        "acc7": round(acc7(y_true, y_pred), 1),
        "acc2": round(acc2_nonzero(y_true, y_pred), 1),
        "f1": round(f1_nonzero(y_true, y_pred), 1),
    }


def unparsable_rate(outputs: Sequence[str]) -> float:
    """Fraction of generative decode strings with no valid number."""
    if not outputs:
        return 0.0
    bad = sum(1 for s in outputs if parse_generative_score(s) is None)
    return round(100.0 * bad / len(outputs), 2)


def out_of_range_rate(
    outputs: Sequence[str],
    *,
    lo: float = -3.0,
    hi: float = 3.0,
) -> float:
    if not outputs:
        return 0.0
    oob = 0
    for s in outputs:
        val = parse_generative_score(s)
        if val is not None and (val < lo or val > hi):
            oob += 1
    return round(100.0 * oob / len(outputs), 2)


def parse_generative_score(text: str) -> float | None:
    """Robust parser for generative readout strings."""
    import re

    s = text.strip().lower()
    if s in ("positive", "pos"):
        return 1.5
    if s in ("negative", "neg"):
        return -1.5
    if s in ("neutral", "neu"):
        return 0.0
    match = re.search(r"-?\d+\.?\d*", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_labels(y: Sequence[float], mu: float, sigma: float) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if sigma < 1e-12:
        return arr - mu
    return (arr - mu) / sigma


def denormalize_labels(y_norm: Sequence[float], mu: float, sigma: float) -> np.ndarray:
    return np.asarray(y_norm, dtype=np.float64) * sigma + mu


def knn_label_smoothness(
    embeddings: np.ndarray,
    labels: Sequence[float],
    *,
    k: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """kNN absolute label difference vs permuted baseline (Section 4.7)."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels, dtype=np.float64)
    n = lab.shape[0]
    if n <= k:
        return {"knn_diff": 0.0, "permuted_baseline": 0.0, "reduction_pct": 0.0}
    # pairwise L2 distances
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
    nn_diff = float(np.mean(np.abs(lab[:, None] - lab[nn_idx])))
    perm = rng.permutation(lab)
    perm_diff = float(np.mean(np.abs(perm[:, None] - perm[nn_idx])))
    reduction = 100.0 * (1.0 - nn_diff / perm_diff) if perm_diff > 0 else 0.0
    return {
        "knn_diff": round(nn_diff, 3),
        "permuted_baseline": round(perm_diff, 3),
        "reduction_pct": round(reduction, 1),
    }
