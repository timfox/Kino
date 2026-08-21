"""Poisson → Laplace graph inference (Eq. 5–10, Theorem 5.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def matched_poisson_rate(u: Tensor, delta: Tensor) -> Tensor:
    """m_ij from Gaussian proxy — Eq. (5)."""
    return (2.0 * u - 1.0 + torch.sqrt((2.0 * u - 1.0).pow(2) + 8.0 * delta.pow(2) + 1e-8)) / 4.0


def poisson_kl(m: Tensor, m0: Tensor) -> Tensor:
    """KL(Pois(m) || Pois(m0)) — Eq. (7)."""
    m = m.clamp(min=1e-6)
    m0 = m0.clamp(min=1e-6)
    return m0 - m + m * (torch.log(m) - torch.log(m0))


def laplace_kl(b: Tensor, b0: float) -> Tensor:
    """KL(Lap(0,b) || Lap(0,b0)) — Eq. (15)."""
    b = b.clamp(min=1e-6)
    return torch.log(torch.tensor(b0, device=b.device, dtype=b.dtype) / b) + b / b0 - 1.0


def edge_features(h_i: Tensor, h_j: Tensor) -> Tensor:
    """e_ij = MLP([hi; hj; |hi-hj|; hi⊙hj]) — Sec. 5.2."""
    return torch.cat([h_i, h_j, (h_i - h_j).abs(), h_i * h_j], dim=-1)


class PoissonLaplaceGraphBranch(nn.Module):
    """NPN-style Poisson prior/posterior + Laplace sparsification stub."""

    def __init__(self, node_dim: int, edge_hidden: int = 256) -> None:
        super().__init__()
        in_dim = node_dim * 4
        self.prior_net = nn.Sequential(nn.Linear(in_dim, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, 1))
        self.post_net = nn.Sequential(
            nn.Linear(in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 2),
        )
        self.lap_scale_net = nn.Sequential(nn.Linear(in_dim + 1, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, 1))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h: [B, K, D] -> adjacency [B, K, K], Poisson KL, Laplace KL."""
        bsz, k, _ = h.shape
        adj = h.new_zeros(bsz, k, k)
        pois_kl = h.new_zeros(())
        lap_kl = h.new_zeros(())
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                e = edge_features(h[:, i], h[:, j])
                m0 = torch.nn.functional.softplus(self.prior_net(e)).squeeze(-1)
                post = self.post_net(e)
                u, log_delta = post[:, 0], post[:, 1]
                delta = torch.nn.functional.softplus(log_delta) + 1e-4
                m = matched_poisson_rate(u, delta)
                pois_kl = pois_kl + poisson_kl(m, m0).mean()

                lap_in = torch.cat([e, u.unsqueeze(-1)], dim=-1)
                bij = torch.nn.functional.softplus(self.lap_scale_net(lap_in)).squeeze(-1) + 1e-4
                lap_kl = lap_kl + laplace_kl(bij, 1.0).mean()

                w = torch.relu(u)
                adj[:, i, j] = w
                adj[:, j, i] = w
                count += 1
        if count:
            pois_kl = pois_kl / count
            lap_kl = lap_kl / count
        return adj, pois_kl, lap_kl
