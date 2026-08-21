"""Linear GCN for node classification (Shi et al. [20], Keriven [18])."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gnn_gen.convolution import symmetric_normalized_adjacency


def linear_gcn_forward(
    x: Tensor,
    adj: Tensor,
    weights: list[Tensor],
    *,
    s: Tensor | None = None,
) -> Tensor:
    """H^(L) = S^L X W_0 ... W_{L-1} with identity activation (Section 3 oversmoothing)."""
    if s is None:
        s = symmetric_normalized_adjacency(adj)
    h = x
    for w in weights:
        h = s @ h @ w
    return h


def linear_gcn_predict(
    x: Tensor,
    adj: Tensor,
    weights: list[Tensor],
    beta: Tensor,
) -> Tensor:
    """Node scores f(i) = H^(L)_i beta."""
    h = linear_gcn_forward(x, adj, weights)
    return h @ beta.view(-1, 1).squeeze(-1)


def oversmoothing_rank_one_limit(
    adj: Tensor,
    n_powers: int,
) -> Tensor:
    """S^n tends to rank-one for symmetric normalised S; returns S^n for analysis."""
    s = symmetric_normalized_adjacency(adj)
    out = s
    for _ in range(n_powers - 1):
        out = s @ out
    return out
