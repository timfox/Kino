"""Minimal EGNN stub with P-EGNN perturbations (Sec. 3.1, 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.pmlip.noise import NoiseGenerator, PerturbedMLP


class EGNNLayer(nn.Module):
    """Single EGNN layer: perturbed edge/node MLPs, clean position update."""

    def __init__(self, hidden: int, noise_dim: int) -> None:
        super().__init__()
        self.edge_mlp = PerturbedMLP(hidden * 2 + 3, hidden, hidden, noise_dim)
        self.node_mlp = PerturbedMLP(hidden * 2, hidden, hidden, noise_dim)
        self.pos_mlp = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        rel = x[dst] - x[src]
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sq_dist = dist * dist
        charge_prod = (h[src, 0:1] * h[dst, 0:1])
        edge_feat = torch.cat([h[src], h[dst], sq_dist, dist, charge_prod], dim=-1)
        msg = self.edge_mlp(edge_feat, z)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1), z)
        coef = self.pos_mlp(h[dst])
        dx = coef * (rel / dist)
        x = x.clone()
        x.index_add_(0, dst, dx)
        return h, x


class EGNNBackbone(nn.Module):
    def __init__(self, *, in_dim: int = 4, hidden: int = 64, layers: int = 4, noise_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden, noise_dim) for _ in range(layers)])
        self.out = nn.Linear(hidden, 3)
        self.noise_gen = NoiseGenerator(noise_dim)

    def forward_once(self, h: Tensor, x: Tensor, edge_index: Tensor, z: Tensor) -> Tensor:
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index, z)
        return self.out(h)

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor, *, k: int = 1) -> Tensor:
        """Return position deltas ``(K, N, 3)``."""
        outs = []
        for _ in range(k):
            z = self.noise_gen.sample()[0]
            outs.append(self.forward_once(h, x, edge_index, z))
        return torch.stack(outs, dim=0)


class PEGNN(nn.Module):
    """P-EGNN for N-body benchmark."""

    def __init__(self, hidden: int = 64, layers: int = 4, noise_dim: int = 32) -> None:
        super().__init__()
        self.backbone = EGNNBackbone(in_dim=4, hidden=hidden, layers=layers, noise_dim=noise_dim)

    def predict(self, h: Tensor, x: Tensor, edge_index: Tensor, *, k: int = 10) -> Tensor:
        return self.backbone(h, x, edge_index, k=k)

    @property
    def noise_dim(self) -> int:
        return self.backbone.noise_gen.wz.in_features

    def extra_param_ratio(self) -> float:
        """Approximate ~13% overhead vs deterministic EGNN (Appendix A)."""
        total = sum(p.numel() for p in self.parameters())
        noise_only = sum(p.numel() for n, p in self.named_parameters() if "noise" in n or "wz" in n)
        return noise_only / max(total - noise_only, 1)
