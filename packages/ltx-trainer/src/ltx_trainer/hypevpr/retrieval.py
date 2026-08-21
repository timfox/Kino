"""Adjustable hierarchical retrieval (Sec. 4.6, Eq. 18–20)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ltx_trainer.hypevpr.poincare import poincare_distance


def recall_at_k(ranks: Tensor, k: int) -> float:
    """Fraction with correct match in top-k (rank 0 = best)."""
    return float((ranks < k).float().mean().item())


def coarse_retrieve(
    hq: Tensor,
    db_tops: Tensor,
    *,
    c: float,
    top_k_prime: int,
) -> tuple[Tensor, Tensor]:
    """
    First-stage retrieval on h^(1,1) (Eq. 3).
    db_tops [N, D], hq [D] or [1, D].
    Returns (indices, distances).
    """
    if hq.dim() == 1:
        hq = hq.unsqueeze(0)
    dists = poincare_distance(hq.expand(db_tops.shape[0], -1), db_tops, c=c)
    k = min(top_k_prime, dists.shape[0])
    vals, idx = torch.topk(dists, k=k, largest=False)
    return idx.view(-1), vals.view(-1)


def level_min_distance(hq: Tensor, level_descs: list[Tensor], *, c: float) -> float:
    """d_ℓ = min_k d_c(h_q, h^(ℓ,k)) (Eq. 18)."""
    if not level_descs:
        return 0.0
    stacked = torch.stack(level_descs, dim=0)
    if hq.dim() == 1:
        hq = hq.unsqueeze(0)
    dists = poincare_distance(hq.expand(stacked.shape[0], -1), stacked, c=c)
    return float(dists.min().item())


def zscore_scores(raw: list[float], eps: float = 1e-6) -> list[float]:
    """ŝ_ℓ from Eq. 19 (negative distance → higher is better)."""
    if not raw:
        return []
    t = torch.tensor(raw, dtype=torch.float32)
    mu = t.mean()
    sigma = t.std(unbiased=False).clamp_min(eps)
    normed = -(t - mu) / sigma
    return [float(x) for x in normed]


def hierarchical_rerank_score(
    hq: Tensor,
    db_tree: dict[int, list[Tensor]],
    *,
    levels: Sequence[int],
    weights: dict[int, float],
    c: float,
) -> float:
    """Weighted sum over selected levels (Eq. 20)."""
    score = 0.0
    for level in levels:
        if level not in db_tree:
            continue
        d_l = level_min_distance(hq, db_tree[level], c=c)
        w = weights.get(level, 1.0)
        score += w * (-d_l)
    return score


def rerank_candidates(
    hq: Tensor,
    candidate_indices: Tensor,
    db_trees: list[dict[int, list[Tensor]]],
    *,
    levels: Sequence[int],
    weights: dict[int, float],
    c: float,
) -> Tensor:
    """Sort candidates by fused hierarchical score."""
    scores: list[float] = []
    for idx in candidate_indices.tolist():
        scores.append(
            hierarchical_rerank_score(
                hq,
                db_trees[idx],
                levels=levels,
                weights=weights,
                c=c,
            )
        )
    order = torch.tensor(scores).argsort(descending=True)
    return candidate_indices[order]
