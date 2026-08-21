"""N-body-style synthetic particle graphs."""

from __future__ import annotations

import torch
from torch import Tensor


def _chain_edges(n: int) -> Tensor:
    if n < 2:
        return torch.zeros(2, 0, dtype=torch.long)
    src = torch.arange(n - 1)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def synthesize_nbody(
    *,
    n_particles: int = 8,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(h, x, edge_index, target_delta)`` for charged N-body stub."""
    g = torch.Generator().manual_seed(seed)
    charges = torch.randn(n_particles, generator=g) * 0.5
    vel = torch.randn(n_particles, 3, generator=g) * 0.1
    pos = torch.randn(n_particles, 3, generator=g)
    h = torch.stack([charges, vel.norm(dim=-1), torch.zeros(n_particles), torch.ones(n_particles)], dim=-1)
    target = pos + vel * 0.5 + torch.randn(n_particles, 3, generator=g) * 0.05
    return h, pos, _chain_edges(n_particles), target - pos
