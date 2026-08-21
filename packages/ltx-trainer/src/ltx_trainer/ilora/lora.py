"""Static LoRA vs graph hypernetwork — Eq. (1), (20)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StaticLoRA(nn.Module):
    """W = W0 + s BA with fixed A, B — Sec. 3."""

    def __init__(self, d_in: int, d_out: int, rank: int, *, alpha: int) -> None:
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(d_out, rank))

    def delta_weight(self) -> Tensor:
        return self.scale * (self.lora_b @ self.lora_a)

    def forward(self, x: Tensor, w0: Tensor) -> Tensor:
        return x @ (w0 + self.delta_weight()).T


class GraphHyperLoRA(nn.Module):
    """Generate input-conditioned A from h_graph; B static — Eq. (20)."""

    def __init__(self, graph_dim: int, d_in: int, d_out: int, rank: int, *, alpha: int) -> None:
        super().__init__()
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out
        self.scale = alpha / rank
        self.lora_b = nn.Parameter(torch.zeros(d_out, rank))
        self.proj_a = nn.Linear(graph_dim, rank * d_in)

    def generate_a(self, h_graph: Tensor) -> Tensor:
        return self.proj_a(h_graph).view(-1, self.rank, self.d_in)

    def forward(self, x: Tensor, h_graph: Tensor, w0: Tensor) -> Tensor:
        a = self.generate_a(h_graph)
        b = self.lora_b.unsqueeze(0).expand(a.shape[0], -1, -1)
        delta = self.scale * torch.bmm(b, a)
        w = w0.unsqueeze(0) + delta
        return torch.bmm(x.unsqueeze(1), w.transpose(1, 2)).squeeze(1)
