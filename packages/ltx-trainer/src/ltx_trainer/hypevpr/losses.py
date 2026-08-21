"""Training objectives L_hier + L_hyp + L_euc (Sec. 4.7, Eq. 21–24)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hypevpr.poincare import logmap0, poincare_distance


def euclidean_triplet_l2(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float,
) -> Tensor:
    da = (anchor - positive).pow(2).sum(dim=-1)
    dn = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(da - dn + margin).mean()


def hyperbolic_triplet_loss(
    hq: Tensor,
    h_pos: Tensor,
    h_neg: Tensor,
    *,
    margin: float,
    c: float,
) -> Tensor:
    """L_hyp (Eq. 22)."""
    dp = poincare_distance(hq, h_pos, c=c)
    dn = poincare_distance(hq, h_neg, c=c)
    return F.relu(dp - dn + margin).mean()


def euclidean_triplet_loss(
    hq: Tensor,
    h_pos_leaf: Tensor,
    h_neg_leaf: Tensor,
    *,
    margin: float,
    c: float,
) -> Tensor:
    """L_euc on log-mapped descriptors (Eq. 23)."""
    dq = logmap0(hq, c=c)
    d_pos = logmap0(h_pos_leaf, c=c)
    d_neg = logmap0(h_neg_leaf, c=c)
    return euclidean_triplet_l2(dq, d_pos, d_neg, margin)


def hierarchical_triplet_loss(
    by_level: dict[int, list[Tensor]],
    *,
    margin: float,
    c: float,
) -> Tensor:
    """
    Simplified L_hier (Eq. 21): parent closer to child than unrelated sibling.
    """
    losses: list[Tensor] = []
    levels = sorted(by_level.keys(), reverse=True)
    for level in levels:
        if level <= 1:
            continue
        parents = by_level.get(level - 1, [])
        children = by_level[level]
        if not parents or not children:
            continue
        parent = parents[0]
        for j, child in enumerate(children):
            neg_idx = (j + 1) % len(children)
            neg = children[neg_idx]
            dp = poincare_distance(parent, child, c=c)
            dn = poincare_distance(child, neg, c=c)
            losses.append(F.relu(dp - dn + margin))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def total_loss(
    hq: Tensor,
    by_level: dict[int, list[Tensor]],
    h_pos: Tensor,
    h_neg: Tensor,
    *,
    margin: float,
    c: float,
) -> dict[str, Tensor]:
    top = by_level[1][0]
    leaves = by_level[max(by_level.keys())]
    h_pos_leaf = leaves[0]
    h_neg_leaf = leaves[min(1, len(leaves) - 1)]
    lh = hierarchical_triplet_loss(by_level, margin=margin, c=c)
    lhyp = hyperbolic_triplet_loss(hq, h_pos, h_neg, margin=margin, c=c)
    leuc = euclidean_triplet_loss(hq, h_pos_leaf, h_neg_leaf, margin=margin, c=c)
    return {"L_hier": lh, "L_hyp": lhyp, "L_euc": leuc, "L_total": lh + lhyp + leuc}
