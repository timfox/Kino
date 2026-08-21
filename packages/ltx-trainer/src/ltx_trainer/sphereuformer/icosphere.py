"""Icosphere discretization (Tab. 1, Sec. 3)."""

from __future__ import annotations

import torch
from torch import Tensor

_GRAPH_CACHE: dict[tuple[int, int], Tensor] = {}


def icosphere_faces(rank: int) -> int:
    return 20 * (4**rank)


def icosphere_vertices(rank: int) -> int:
    return 10 * (4**rank) + 2


def node_count(rank: int, *, node_type: str) -> int:
    if node_type == "ico":
        return icosphere_faces(rank)
    return icosphere_vertices(rank)


def k_order_neighbors(n_nodes: int, *, c_win: int, use_cache: bool = True) -> Tensor:
    """Neighbor indices for SLSA (K-order ring on fixed graph stub)."""
    key = (n_nodes, c_win)
    if use_cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    base = torch.arange(n_nodes)
    rings = [base]
    for k in range(1, c_win + 1):
        rings.append((base + k) % n_nodes)
        rings.append((base - k) % n_nodes)
    nbr = torch.stack(rings, dim=0)  # (1+2*c_win, N)
    if use_cache:
        _GRAPH_CACHE[key] = nbr
    return nbr


def center_pool(f: Tensor) -> Tensor:
    """Downsample: keep every 4th node (rank subdivision stub)."""
    n = f.shape[1]
    if n < 4:
        return f
    return f[:, : n // 4 * 4].view(f.shape[0], -1, 4, f.shape[-1]).mean(dim=2)


def nearest_up(f: Tensor, target_n: int) -> Tensor:
    """Upsample by repeating lower-res nodes."""
    b, n, c = f.shape
    if n >= target_n:
        return f[:, :target_n]
    reps = (target_n + n - 1) // n
    up = f.repeat_interleave(reps, dim=1)[:, :target_n]
    return up
