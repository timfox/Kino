"""FastNBSim-style quasi-clique search + credit-based dynamic updates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def edge_density(vertices: Iterable[int], adj: dict[int, set[int]]) -> float:
    verts = sorted(set(vertices))
    n = len(verts)
    if n <= 1:
        return 1.0
    edges = 0
    for i, u in enumerate(verts):
        for v in verts[i + 1 :]:
            if v in adj.get(u, set()):
                edges += 1
    return (2.0 * edges) / (n * (n - 1))


def neighborhood_containment(u: int, v: int, adj: dict[int, set[int]]) -> float:
    nu = adj.get(u, set())
    nv = adj.get(v, set())
    if not nu:
        return 0.0
    return len(nu & nv) / len(nu)


def gamma_degree(u: int, adj: dict[int, set[int]], gamma: float) -> int:
    nu = adj.get(u, set())
    if not nu:
        return 0
    return sum(1 for v in nu if len(adj.get(v, set())) >= gamma * len(nu))


def fast_nbsim_seed(
    adj: dict[int, set[int]],
    *,
    alpha: float = 0.8,
    gamma: float = 0.5,
) -> list[int]:
    """Greedy expand from highest γ-degree seed (toy FastNBSim)."""
    if not adj:
        return []
    order = sorted(adj.keys(), key=lambda u: gamma_degree(u, adj, gamma), reverse=True)
    best: list[int] = []
    for seed in order[: min(5, len(order))]:
        cand = [seed]
        pool = sorted(
            adj.get(seed, set()),
            key=lambda v: neighborhood_containment(seed, v, adj),
            reverse=True,
        )
        for v in pool:
            trial = cand + [v]
            if edge_density(trial, adj) >= alpha:
                cand = trial
        if len(cand) > len(best):
            best = cand
    return best


class CreditTracker:
    """Per-vertex credits for incremental quasi-clique maintenance."""

    def __init__(self, vertices: Iterable[int], *, initial: float = 1.0) -> None:
        self.credits: dict[int, float] = {v: initial for v in vertices}

    def apply_edge(self, u: int, v: int, *, inserted: bool, delta: float = 0.05) -> None:
        for x in (u, v):
            if x not in self.credits:
                self.credits[x] = 1.0
        sign = 1.0 if inserted else -1.0
        self.credits[u] = max(0.0, self.credits[u] + sign * delta)
        self.credits[v] = max(0.0, self.credits[v] + sign * delta)

    def active(self, threshold: float = 0.5) -> list[int]:
        return sorted(v for v, c in self.credits.items() if c >= threshold)


def dynamic_update_smoke(
    adj: dict[int, set[int]],
    updates: list[tuple[int, int, bool]],
    *,
    alpha: float = 0.8,
    gamma: float = 0.5,
) -> dict[str, Any]:
    """Replay edge updates; re-seed when credits drop."""
    tracker = CreditTracker(adj.keys())
    sizes: list[int] = []
    for u, v, ins in updates:
        if ins:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        else:
            adj.get(u, set()).discard(v)
            adj.get(v, set()).discard(u)
        tracker.apply_edge(u, v, inserted=ins)
        clique = fast_nbsim_seed(adj, alpha=alpha, gamma=gamma)
        sizes.append(len(clique))
    return {"final_size": sizes[-1] if sizes else 0, "size_trace": sizes}


def toy_dynamic_graph(seed: int = 0) -> dict[int, set[int]]:
    rng = np.random.default_rng(seed)
    n = 20
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    core = list(range(8))
    for i in core:
        for j in core:
            if i < j:
                adj[i].add(j)
                adj[j].add(i)
    for _ in range(12):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj
