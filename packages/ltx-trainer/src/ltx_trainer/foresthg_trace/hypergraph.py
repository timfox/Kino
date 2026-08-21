"""Multimodal ecological scene hypergraph H = (V, E, X)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

HyperedgeFamily = Literal["BEH", "REH", "CEH"]


@dataclass
class TreeNode:
    node_id: int
    health: float
    height_m: float
    slope_class: str  # shady | sunny
    density_class: str = "medium"
    species_group: str = "mixed"
    x: float = 0.0
    y: float = 0.0
    attrs: dict[str, float] = field(default_factory=dict)


@dataclass
class Hyperedge:
    edge_id: str
    family: HyperedgeFamily
    node_ids: tuple[int, ...]
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneHypergraph:
    scene_id: str
    nodes: dict[int, TreeNode]
    hyperedges: list[Hyperedge]

    def node_list(self) -> list[TreeNode]:
        return list(self.nodes.values())

    def filter_nodes(self, **predicates: Any) -> list[TreeNode]:
        out: list[TreeNode] = []
        for n in self.nodes.values():
            ok = True
            for key, val in predicates.items():
                if getattr(n, key, None) != val:
                    ok = False
                    break
            if ok:
                out.append(n)
        return out

    def nodes_in_hyperedge(self, edge_id: str) -> list[TreeNode]:
        for e in self.hyperedges:
            if e.edge_id == edge_id:
                return [self.nodes[i] for i in e.node_ids if i in self.nodes]
        return []
