"""Shared-importance synchronization for model-side pruning (Sec. III-C2)."""

from __future__ import annotations

import numpy as np


def aggregate_receptive_field(
    s: np.ndarray,
    *,
    kernel: int = 3,
) -> np.ndarray:
    """
    Pool importance scores within each filter receptive field.

    Returns per-filter scalar importances ``I`` (length K ≈ number of filter slots).
    """
    score = np.asarray(s, dtype=np.float64)
    if score.ndim != 2:
        raise ValueError("S must be H×W")
    h, w = score.shape
    k = max(1, kernel)
    bh = max(1, h // k)
    bw = max(1, w // k)
    importances: list[float] = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = score[i : min(i + bh, h), j : min(j + bw, w)]
            importances.append(float(np.mean(block)))
    return np.asarray(importances, dtype=np.float64)


def model_mask_from_importance(
    importances: np.ndarray,
    *,
    sw: float,
    tau: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Structured model mask: keep top ``(1-sw)*K`` filters by aggregated importance.

    Returns ``(M_weight, tau_used)`` with ``M_weight`` in {0,1}^K.
    """
    i_arr = np.asarray(importances, dtype=np.float64)
    k = i_arr.size
    if k == 0:
        return np.zeros(0, dtype=np.float64), 0.0
    keep = max(1, int(round((1.0 - sw) * k)))
    order = np.argsort(-i_arr)
    mask = np.zeros(k, dtype=np.float64)
    mask[order[:keep]] = 1.0
    if tau is None:
        tau_used = float(i_arr[order[keep - 1]]) if keep <= k else 0.0
    else:
        tau_used = tau
        mask = (i_arr >= tau_used).astype(np.float64)
    return mask, tau_used


def synchronized_masks(
    s: np.ndarray,
    *,
    sd: float = 0.7,
    sw: float = 0.7,
    ste_threshold: float = 0.5,
    kernel: int = 3,
) -> dict[str, np.ndarray | float]:
    """Joint data + model masks from shared importance field ``S``."""
    from ltx_trainer.dyna_pruner.masks import hard_mask_ste, soft_data_mask

    soft = soft_data_mask(s)
    # Keep top (1-sd) fraction by importance; prune the rest.
    flat = np.sort(soft.ravel())
    idx = min(len(flat) - 1, max(0, int(round(sd * len(flat)))))
    thresh = float(flat[idx]) if len(flat) else ste_threshold
    m_data = hard_mask_ste(soft, threshold=thresh)
    importances = aggregate_receptive_field(s, kernel=kernel)
    m_weight, tau = model_mask_from_importance(importances, sw=sw)
    return {
        "S": soft,
        "M_data": m_data,
        "M_weight": m_weight,
        "I": importances,
        "tau": tau,
        "data_threshold": thresh,
    }
