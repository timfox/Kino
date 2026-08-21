"""Metaphorical knowledge graph and MetaphorBoost querying (Eq. 4–6)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class MetaphorKnowledgeGraph:
    """Directed multi-edge graph over metaphor concept pairs (cs, ct)."""

    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> MetaphorKnowledgeGraph:
        g = cls()
        for src, tgt in pairs:
            g.add_edge(src.lower().strip(), tgt.lower().strip())
        return g

    def add_edge(self, src: str, tgt: str) -> None:
        self.edges[src].add(tgt)
        self.reverse[tgt].add(src)

    def neighbors_within_hops(self, node: str, hops: int) -> set[str]:
        """``N^h_G(k)`` — nodes reachable within ``h`` hops."""
        node = node.lower().strip()
        if node not in self.edges and node not in self.reverse:
            return set()
        frontier = {node}
        visited = {node}
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                nxt |= self.edges.get(n, set())
                nxt |= self.reverse.get(n, set())
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        visited.discard(node)
        return visited

    def degree_to_keywords(self, candidate: str, keywords: set[str]) -> int:
        """``deg(·, K)`` — edges linking candidate to any keyword neighborhood."""
        cand = candidate.lower().strip()
        score = 0
        for kw in keywords:
            if cand == kw:
                score += 2
            if cand in self.edges.get(kw, set()) or kw in self.edges.get(cand, set()):
                score += 1
            if cand in self.reverse.get(kw, set()) or kw in self.reverse.get(cand, set()):
                score += 1
        return score


def top_z_references(
    keywords: list[str],
    graph: MetaphorKnowledgeGraph,
    *,
    hops: int = 2,
    z: int = 10,
) -> list[str]:
    """Eq. (5): retain top-``z`` targets with highest keyword connectivity."""
    kw_set = {k.lower().strip() for k in keywords}
    pool: set[str] = set()
    for kw in kw_set:
        pool |= graph.neighbors_within_hops(kw, hops)
    ranked = sorted(pool, key=lambda c: graph.degree_to_keywords(c, kw_set), reverse=True)
    return ranked[:z]


def identify_keywords_stub(elements: list[str]) -> list[str]:
    """Stub for ``K = Identify(v ⊕ t)`` (Eq. 4)."""
    return [e.lower().strip() for e in elements if e.strip()]
