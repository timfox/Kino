"""Graph convolution readout — Eq. (19)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def normalize_adjacency(adj: Tensor) -> Tensor:
    """D^{-1/2} Ã D^{-1/2} with self-loops."""
    adj = adj + torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype).unsqueeze(0)
    deg = adj.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)


class GraphEncoder(nn.Module):
    """Two-layer GCN + mean pool -> h_graph."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h: Tensor, adj: Tensor) -> Tensor:
        a = normalize_adjacency(adj)
        x = torch.relu(torch.matmul(a, self.w1(h)))
        x = torch.matmul(a, self.w2(x))
        return x.mean(dim=1)
