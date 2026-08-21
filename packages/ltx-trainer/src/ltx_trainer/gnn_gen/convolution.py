"""Graph convolutions S used in GCN (2) and graphon limit (Section 3)."""

from __future__ import annotations

import torch
from torch import Tensor


def symmetric_normalized_adjacency(adj: Tensor, *, add_self_loops: bool = True) -> Tensor:
    """S_sym = (D+I)^{-1/2} (A+I) (D+I)^{-1/2} (Kipf & Welling GCN)."""
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be (n, n)")
    a = adj.clone()
    if add_self_loops:
        a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1).clamp(min=1e-12)
    inv_sqrt = deg.pow(-0.5)
    d_left = torch.diag(inv_sqrt)
    d_right = torch.diag(inv_sqrt)
    return d_left @ a @ d_right


def row_normalized_adjacency(adj: Tensor, *, add_self_loops: bool = True) -> Tensor:
    """S_row = (D+I)^{-1} (A+I)."""
    a = adj.clone()
    if add_self_loops:
        a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1).clamp(min=1e-12)
    return torch.diag(deg.reciprocal()) @ a


def max_degree(adj: Tensor) -> int:
    """Maximum node degree (unweighted edges > 0)."""
    deg = (adj > 0).sum(dim=1)
    return int(deg.max().item())
