"""Proximity-graph stubs for Yi vector-level updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphVertex:
    vid: int
    neighbors: list[int] = field(default_factory=list)
    deleted: bool = False
    raw: tuple[float, ...] | None = None


@dataclass
class ProximityGraph:
    """Singly-linked on-disk style ANN graph (in-memory stub)."""

    vertices: dict[int, GraphVertex] = field(default_factory=dict)
    entry: int = 0
    max_degree: int = 96

    def add(self, vid: int, neighbors: list[int] | None = None, raw: tuple[float, ...] | None = None) -> None:
        self.vertices[vid] = GraphVertex(vid=vid, neighbors=list(neighbors or []), raw=raw)

    def mark_deleted(self, vid: int) -> None:
        if vid in self.vertices:
            self.vertices[vid].deleted = True

    def alive(self) -> list[int]:
        return [v for v, n in self.vertices.items() if not n.deleted]


def demo_graph(n: int = 32, degree: int = 4) -> ProximityGraph:
    """Small ring+shortcut graph for CPU demos."""
    g = ProximityGraph(entry=0, max_degree=degree)
    for i in range(n):
        nbrs = [(i + 1) % n, (i + 3) % n, (i + 7) % n]
        g.add(i, nbrs[:degree], raw=(float(i), float(i % 5)))
    return g


def beam_search(graph: ProximityGraph, query: tuple[float, ...], k: int = 5) -> list[int]:
    """Greedy beam-1 search using L2 on optional raw coords (proxy)."""
    if not graph.vertices:
        return []
    start = graph.entry if graph.entry in graph.vertices else next(iter(graph.vertices))
    visited: set[int] = set()
    cand = start
    best: list[tuple[float, int]] = []

    def dist(vid: int) -> float:
        v = graph.vertices[vid]
        if v.raw is None or not query:
            return float(abs(vid - int(query[0])) if query else vid)
        return sum((a - b) ** 2 for a, b in zip(v.raw, query, strict=False)) ** 0.5

    while cand not in visited:
        visited.add(cand)
        node = graph.vertices[cand]
        if not node.deleted:
            best.append((dist(cand), cand))
            best.sort()
            best = best[:k]
        nxt = None
        best_d = dist(cand)
        for nb in node.neighbors:
            if nb in visited or nb not in graph.vertices or graph.vertices[nb].deleted:
                continue
            d = dist(nb)
            if d < best_d:
                best_d = d
                nxt = nb
        if nxt is None:
            break
        cand = nxt
    return [vid for _, vid in best]


def prune_neighbors(graph: ProximityGraph, vid: int, candidates: list[int]) -> list[int]:
    """Degree-bounded prune (simplified DiskANN-style keep-first)."""
    uniq = []
    seen = set()
    for c in candidates:
        if c == vid or c in seen or c not in graph.vertices or graph.vertices[c].deleted:
            continue
        seen.add(c)
        uniq.append(c)
        if len(uniq) >= graph.max_degree:
            break
    return uniq
