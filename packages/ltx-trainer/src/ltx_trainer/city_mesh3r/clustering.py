"""SLPA overlapping communities for distributed SfM (Sec. 3.1)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def slpa_overlapping_communities(
    adj: Tensor,
    *,
    iterations: int = 20,
    membership_threshold: float = 0.1,
    min_overlap: int = 2,
) -> list[set[int]]:
    """Simplified Speaker–Listener Label Propagation on an adjacency matrix."""
    n = adj.shape[0]
    if n == 0:
        return []
    labels = list(range(n))
    memory: list[list[int]] = [[i] * 4 for i in range(n)]

    for _ in range(iterations):
        order = torch.randperm(n).tolist()
        for i in order:
            neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0].tolist()
            if not neighbors:
                continue
            j = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
            counts: dict[int, int] = {}
            for lab in memory[j]:
                counts[lab] = counts.get(lab, 0) + 1
            speaker_label = max(counts, key=counts.get)
            memory[j].append(speaker_label)
            if len(memory[j]) > 8:
                memory[j] = memory[j][-8:]

    communities: dict[int, set[int]] = {}
    for i in range(n):
        counts: dict[int, int] = {}
        for lab in memory[i]:
            counts[lab] = counts.get(lab, 0) + 1
        total = sum(counts.values()) or 1
        for lab, c in counts.items():
            if c / total >= membership_threshold:
                communities.setdefault(lab, set()).add(i)

    clusters = [c for c in communities.values() if len(c) >= 2]
    if not clusters:
        return [set(range(n))]
    return clusters


def cluster_overlap_matrix(clusters: list[set[int]], n_images: int) -> Tensor:
    """|Ci ∩ Cj| weights for cluster graph."""
    m = len(clusters)
    w = torch.zeros(m, m)
    for i in range(m):
        for j in range(i + 1, m):
            ov = len(clusters[i] & clusters[j])
            w[i, j] = w[j, i] = float(ov)
    return w


def minimum_spanning_tree_edges(weights: Tensor) -> list[tuple[int, int, float]]:
    """Kruskal MST on symmetric weight matrix (zero diagonal)."""
    m = weights.shape[0]
    edges: list[tuple[int, int, float]] = []
    for i in range(m):
        for j in range(i + 1, m):
            if weights[i, j] > 0:
                edges.append((i, j, float(weights[i, j].item())))
    edges.sort(key=lambda e: e[2])
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst: list[tuple[int, int, float]] = []
    for i, j, w in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j, w))
        if len(mst) >= m - 1:
            break
    return mst
