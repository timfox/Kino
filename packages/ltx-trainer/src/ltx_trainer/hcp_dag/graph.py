"""Planar crack graph → crack chains and child→parent DAG (Sec. II)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.hcp_dag.config import HCPGraphRules


@dataclass(frozen=True)
class Edge:
    u: int
    v: int

    def nodes(self) -> tuple[int, int]:
        return (self.u, self.v)

    def other(self, node: int) -> int:
        return self.v if node == self.u else self.u


@dataclass
class CrackGraph:
    """Embedded graph: nodes are junctions/boundary ends, edges are crack segments."""

    edges: list[Edge]
    positions: dict[int, tuple[float, float]]

    def node_degree(self) -> dict[int, int]:
        deg: dict[int, int] = {}
        for e in self.edges:
            deg[e.u] = deg.get(e.u, 0) + 1
            deg[e.v] = deg.get(e.v, 0) + 1
        return deg

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        adj: dict[int, list[tuple[int, int]]] = {}
        for i, e in enumerate(self.edges):
            for n in e.nodes():
                adj.setdefault(n, []).append((i, e.other(n)))
        return adj


def edge_angle(
    graph: CrackGraph,
    edge_a: int,
    edge_b: int,
    *,
    shared_node: int,
) -> float:
    """Angle in degrees between two edges meeting at shared_node."""
    ea, eb = graph.edges[edge_a], graph.edges[edge_b]
    pa = graph.positions[ea.other(shared_node)]
    pb = graph.positions[eb.other(shared_node)]
    pc = graph.positions[shared_node]
    va = np.array(pa) - np.array(pc)
    vb = np.array(pb) - np.array(pc)
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def same_crack_chain(
    graph: CrackGraph,
    edge_a: int,
    edge_b: int,
    node: int,
    rules: HCPGraphRules,
) -> bool:
    ang = edge_angle(graph, edge_a, edge_b, shared_node=node)
    return rules.min_chain_angle_deg <= ang <= rules.max_chain_angle_deg


def build_crack_chains(graph: CrackGraph, rules: HCPGraphRules | None = None) -> list[list[int]]:
    """Step 1: merge consecutive edges with α ∈ [150°, 180°] into cracks."""
    rules = rules or HCPGraphRules()
    adj = graph.adjacency()
    visited: set[int] = set()
    chains: list[list[int]] = []

    def extend_chain(seed: int) -> list[int]:
        stack = [seed]
        chain: list[int] = []
        while stack:
            eidx = stack.pop()
            if eidx in visited:
                continue
            visited.add(eidx)
            chain.append(eidx)
            e = graph.edges[eidx]
            for node in e.nodes():
                for nxt, _ in adj.get(node, []):
                    if nxt in visited:
                        continue
                    if same_crack_chain(graph, eidx, nxt, node, rules):
                        stack.append(nxt)
        return sorted(chain)

    for i in range(len(graph.edges)):
        if i not in visited:
            chains.append(extend_chain(i))
    return chains


def edge_to_crack(chains: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for cid, chain in enumerate(chains):
        for e in chain:
            mapping[e] = cid
    return mapping


def t_junctions_at_node(
    graph: CrackGraph,
    node: int,
    edge_crack: dict[int, int],
) -> list[tuple[int, int]]:
    """Return (child_crack, parent_crack) for each T at node."""
    adj = graph.adjacency()
    incident = adj.get(node, [])
    if len(incident) != 3:
        return []
    pairs: list[tuple[int, int, float]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            ea, eb = incident[i][0], incident[j][0]
            pairs.append((ea, eb, edge_angle(graph, ea, eb, shared_node=node)))
    # crossbar pair: angle closest to 180°
    ea, eb, _ = max(pairs, key=lambda t: t[2])
    leg_edges = [incident[k][0] for k in range(3) if incident[k][0] not in (ea, eb)]
    if len(leg_edges) != 1:
        return []
    leg_e = leg_edges[0]
    child = edge_crack[leg_e]
    parent = edge_crack[ea]
    if child == parent:
        parent = edge_crack[eb]
    if child == parent:
        return []
    return [(child, parent)]


def build_crack_dag(chains: list[list[int]], graph: CrackGraph) -> dict[str, Any]:
    """Steps 2–3: DAG vertices = cracks, arcs child → parent at T-junctions."""
    n = len(chains)
    edge_crack = edge_to_crack(chains)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    seen: set[tuple[int, int]] = set()

    for node in graph.positions:
        for child, parent in t_junctions_at_node(graph, node, edge_crack):
            if (child, parent) in seen:
                continue
            seen.add((child, parent))
            adj[child].append(parent)

    in_deg = {i: 0 for i in range(n)}
    for child, parents in adj.items():
        for parent in parents:
            in_deg[parent] += 1

    return {"n_cracks": n, "adjacency": adj, "in_degree": in_deg, "chains": chains}


def whole_crack_generations(chains: list[list[int]], gen_by_crack: dict[int, int]) -> dict[int, int]:
    """Map edge index → generation; entire crack shares one label (no fragmentation)."""
    out: dict[int, int] = {}
    for cid, chain in enumerate(chains):
        g = gen_by_crack[cid]
        for e in chain:
            out[e] = g
    return out
