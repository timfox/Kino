"""Weisfeiler–Leman (1-WL) colour refinement vs MP-GNN expressivity (Section 2)."""

from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor


def _multiset_key(label: int, neighbor_labels: list[int]) -> str:
    """Canonical key for (label, sorted neighbor multiset) in 1-WL."""
    return f"{label}|{','.join(str(x) for x in sorted(neighbor_labels))}"


def _refine_colours(colours: list[int], adj: Tensor) -> list[int]:
    """One 1-WL round: map hashed multisets to compact integer colours."""
    n = len(colours)
    keys = []
    for i in range(n):
        nbrs = (adj[i] > 0).nonzero(as_tuple=True)[0].tolist()
        neighbor_labels = [colours[j] for j in nbrs]
        keys.append(_multiset_key(colours[i], neighbor_labels))
    unique = {k: idx for idx, k in enumerate(dict.fromkeys(keys))}
    return [unique[k] for k in keys]


def wl_colors(
    adj: Tensor,
    *,
    initial: int = 0,
    max_iters: int = 32,
) -> tuple[Tensor, int]:
    """Run 1-WL on unweighted graph ``adj`` (n×n, symmetric, non-negative).

    Returns:
        colours: (n,) int64 stable colours after refinement stops.
        n_iters: number of refinement rounds performed.
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square (n, n)")
    n = adj.shape[0]
    device = adj.device
    colours = [initial] * n
    for it in range(max_iters):
        next_colours = _refine_colours(colours, adj)
        if next_colours == colours:
            return torch.tensor(colours, device=device, dtype=torch.long), it
        colours = next_colours
    return torch.tensor(colours, device=device, dtype=torch.long), max_iters


def wl_multiset_signature(adj: Tensor, *, max_iters: int = 32) -> tuple[int, ...]:
    """Multiset of final node colours (graph invariant under 1-WL)."""
    colours, _ = wl_colors(adj, max_iters=max_iters)
    counts = Counter(colours.tolist())
    return tuple(sorted(counts.items()))


def wl_same_under_refinement(adj_a: Tensor, adj_b: Tensor, *, max_iters: int = 32) -> bool:
    """True if 1-WL cannot distinguish the two graphs (same colour histogram)."""
    return wl_multiset_signature(adj_a, max_iters=max_iters) == wl_multiset_signature(
        adj_b, max_iters=max_iters
    )


def cycle_graph(n: int, *, device: torch.device | None = None) -> Tensor:
    """Undirected n-cycle adjacency C_n."""
    if n < 3:
        raise ValueError("cycle needs n >= 3")
    a = torch.zeros(n, n, device=device)
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[(i + 1) % n, i] = 1.0
    return a


def path_graph(n: int, *, device: torch.device | None = None) -> Tensor:
    """Undirected path P_n."""
    if n < 2:
        raise ValueError("path needs n >= 2")
    a = torch.zeros(n, n, device=device)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a
