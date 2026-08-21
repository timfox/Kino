"""NAS task correlation (Fig. 1b motivation)."""

from __future__ import annotations

import numpy as np


# Representative Pearson matrix on curated mouse NAFLD dataset (paper Fig. 1b).
NAS_PEARSON_OURS: np.ndarray = np.array(
    [
        [1.00, 0.62, 0.58],
        [0.62, 1.00, 0.71],
        [0.58, 0.71, 1.00],
    ],
    dtype=np.float64,
)

TASK_LABELS = ("steatosis", "ballooning", "inflammation")


def pearson_matrix(scores: np.ndarray) -> np.ndarray:
    """scores: (n_samples, n_tasks)."""
    if scores.ndim != 2 or scores.shape[0] < 2:
        return np.eye(scores.shape[1] if scores.ndim == 2 else 3)
    corr = np.corrcoef(scores.T)
    return np.nan_to_num(corr, nan=0.0)


def negative_transfer_risk(corr: np.ndarray) -> float:
    """Heuristic: mean off-diagonal |ρ| as interference risk proxy."""
    n = corr.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(corr[mask])))
