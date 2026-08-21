"""Graphon neural network discretisation (Section 3; Keriven et al., Ruiz et al.)."""

from __future__ import annotations

import torch
from torch import Tensor


def graphon_from_adjacency(adj: Tensor) -> Tensor:
    """Treat dense adj/n as empirical graphon sample on grid i/n, j/n."""
    n = adj.shape[0]
    return adj / max(n, 1)


def graphon_degree_function(w: Tensor) -> Tensor:
    """deg(x_i) ≈ (1/n) sum_j W(x_i, x_j) on grid."""
    n = w.shape[0]
    return w.sum(dim=1) / max(n, 1)


def graphon_convolution_apply(
    w: Tensor,
    h: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Discrete S(h)(x_i) = sum_j W_ij h_j / sqrt(deg_i deg_j) (symmetric normalised)."""
    deg = graphon_degree_function(w).clamp(min=eps)
    inv = (deg.unsqueeze(0) * deg.unsqueeze(1)).sqrt().reciprocal()
    return (w * inv) @ h


def graphon_gcn_layer(
    h: Tensor,
    w: Tensor,
    weight: Tensor,
) -> Tensor:
    """One linear graphon-GCN step h' = S(h) W (identity activation)."""
    return graphon_convolution_apply(w, h) @ weight


def discretization_error_bound(n: int) -> float:
    """O(n^{-1/2}) convergence of n-node GNN to graphon NN for dense graphs [33]."""
    return n ** (-0.5)
