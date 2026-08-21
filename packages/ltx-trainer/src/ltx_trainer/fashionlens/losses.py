"""InfoNCE retrieval + PGSQC regularizers — Sec. IV-E, Eq. (11)–(12)."""

from __future__ import annotations

import math


def infonce_retrieval_loss(
    similarities: list[float],
    *,
    temperature: float = 0.07,
    positive_index: int = 0,
) -> float:
    r"""InfoNCE with one positive per row (Eq. 11) — list is logits candidates for same query."""
    if not similarities or positive_index < 0 or positive_index >= len(similarities):
        return 0.0
    logits = [s / temperature for s in similarities]
    m = max(logits)
    log_den = m + math.log(sum(math.exp(x - m) for x in logits))
    return -(logits[positive_index] - log_den)


def total_training_loss(
    l_ret: float,
    l_ortho: float,
    l_reg: float,
    *,
    beta1: float = 1e-2,
    beta2: float = 1e-4,
) -> float:
    r"""L = L_ret + β_1 L_ortho + β_2 L_reg (Eq. 12)."""
    return l_ret + beta1 * l_ortho + beta2 * l_reg


def mean_reciprocal_rank_from_r_at(r1: float, r5: float, r10: float) -> float:
    """mR = (R@1 + R@5 + R@10) / 3 in percentage points (Table II–III)."""
    return (r1 + r5 + r10) / 3.0
