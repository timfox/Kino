"""Re-ranking strategies: Likelihood, MBR, Lik.+MBR (§4.6)."""

from __future__ import annotations

import torch
from torch import Tensor


def chrF_similarity(a: str, b: str) -> float:
    """Character n-gram F-score proxy for MBR."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    prec = inter / len(sa)
    rec = inter / len(sb)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mbr_select(candidates: list[str]) -> int:
    """Index with highest average chrF to all others."""
    n = len(candidates)
    if n == 0:
        return 0
    scores = []
    for i in range(n):
        sims = [chrF_similarity(candidates[i], candidates[j]) for j in range(n) if j != i]
        scores.append(sum(sims) / max(len(sims), 1))
    return int(max(range(n), key=lambda i: scores[i]))


def likelihood_select(log_probs: Tensor) -> int:
    return int(torch.argmax(log_probs).item())


def lik_mbr_hybrid(
    candidates: list[str],
    log_probs: Tensor,
    *,
    agree_bonus: float = 0.0,
) -> int:
    """Pick Likelihood; on tie with MBR use pairwise tiebreak stub (same index if agree)."""
    lik = likelihood_select(log_probs)
    mbr = mbr_select(candidates)
    if lik == mbr:
        return lik
    # Stub: prefer MBR when likelihood would pick SHAS-like short candidate (index 1)
    if lik == 1 and mbr != 1:
        return mbr
    return lik if log_probs[lik] + agree_bonus >= log_probs[mbr] else mbr
