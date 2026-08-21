"""DAS benchmark metrics: accuracy, macro-F1, NAR, FNR."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        m[int(t), int(p)] += 1
    return m


def accuracy(m: np.ndarray) -> float:
    total = m.sum()
    if total == 0:
        return 0.0
    return float(np.trace(m) / total) * 100.0


def macro_f1(m: np.ndarray) -> float:
    c = m.shape[0]
    f1s: list[float] = []
    for i in range(c):
        tp = m[i, i]
        prec_denom = m[:, i].sum()
        rec_denom = m[i, :].sum()
        if prec_denom == 0 or rec_denom == 0:
            continue
        p = tp / prec_denom
        r = tp / rec_denom
        f1s.append(2 * p * r / (p + r))
    if not f1s:
        return 0.0
    return float(np.mean(f1s)) * 100.0


def nar_fnr(m: np.ndarray, background_idx: int = 0) -> tuple[float, float]:
    """Background vs threat aggregation (Eq. 10–11)."""
    threat_mask = np.ones(m.shape[0], dtype=bool)
    threat_mask[background_idx] = False
    tp = int(m[np.ix_(threat_mask, threat_mask)].sum())
    tn = int(m[background_idx, background_idx])
    fp = int(m[background_idx, threat_mask].sum())
    fn = int(m[threat_mask, background_idx].sum())
    nar = 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = 100.0 * fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return nar, fnr
