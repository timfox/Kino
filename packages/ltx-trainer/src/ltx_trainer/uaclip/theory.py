"""Theorem 4.1 — Utility-Aware InfoNCE mutual-information bound (Eq. 5)."""

from __future__ import annotations

import numpy as np


def utility_aware_infonce_bound(
    loss: float,
    n: int,
    visual_utilities: np.ndarray,
    *,
    mutual_information: float | None = None,
    alpha_visual: float = 1.0,
) -> dict[str, float]:
    """
    log N - L <= I(t,v) + α_v (mean(h) - min(h)) when τ=β_s=1 (Theorem 4.1).
    """
    h = np.asarray(visual_utilities, dtype=np.float64)
    utility_term = alpha_visual * (float(h.mean()) - float(h.min()))
    mi = mutual_information if mutual_information is not None else 0.0
    rhs = mi + utility_term
    lhs = float(np.log(max(n, 1)) - loss)
    return {
        "lhs_log_n_minus_loss": lhs,
        "rhs_bound": rhs,
        "mutual_information": mi,
        "utility_spread_term": utility_term,
        "satisfies_bound": lhs <= rhs + 1e-6,
    }
