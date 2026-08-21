"""Dual Containment Index — DC-Index (Sec. IV-B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.kd_truss.truss import online_kd_truss_query


@dataclass
class DcNode:
    k: int
    delta: int
    incremental_edges: set[tuple[int, int]] = field(default_factory=set)
    parent: DcNode | None = None


@dataclass
class DcIndex:
    """Reduced arborescence + compressed lookup table (Fig. 6)."""

    root: DcNode | None = None
    lookup: dict[tuple[int, int], DcNode] = field(default_factory=dict)
    _truss: dict[tuple[int, int], set[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def build(cls, edges: set[tuple[int, int]], k_max: int, delta_max: int) -> DcIndex:
        idx = cls()
        truss: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for k in range(3, k_max + 1):
            for delta in range(delta_max, -1, -1):
                truss[(k, delta)] = online_kd_truss_query(edges, k, delta)
        idx._truss = truss

        nodes: dict[tuple[int, int], DcNode] = {}
        for k in range(3, k_max + 1):
            for delta in range(delta_max, -1, -1):
                nodes[(k, delta)] = DcNode(k=k, delta=delta)

        idx.root = nodes[(k_max, 0)]
        idx.root.incremental_edges = set(truss[(k_max, 0)])

        for k in range(3, k_max + 1):
            for delta in range(delta_max, -1, -1):
                if (k, delta) == (k_max, 0):
                    continue
                cur = truss[(k, delta)]
                parent_key: tuple[int, int] | None = None
                h_diff = 0
                v_diff = 0
                if delta - 1 >= 0:
                    parent_h = (k, delta - 1)
                    h_diff = len(cur - truss[parent_h])
                if k + 1 <= k_max:
                    parent_v = (k + 1, delta)
                    v_diff = len(cur - truss[parent_v])
                if delta - 1 >= 0 and (k + 1 > k_max or h_diff <= v_diff):
                    parent_key = (k, delta - 1)
                elif k + 1 <= k_max:
                    parent_key = (k + 1, delta)

                node = nodes[(k, delta)]
                if parent_key is not None:
                    node.parent = nodes[parent_key]
                    node.incremental_edges = cur - truss[parent_key]

        # Compressed lookup: map (k, δ) to representative with identical truss.
        rep: dict[tuple[int, int], DcNode] = {}
        for k in range(3, k_max + 1):
            last: DcNode | None = None
            for delta in range(delta_max, -1, -1):
                key = (k, delta)
                if last is not None and truss[key] == truss[(last.k, last.delta)]:
                    rep[key] = last
                else:
                    last = nodes[key]
                    rep[key] = last
        idx.lookup = rep
        return idx

    def dc_query(self, k: int, delta: int) -> set[tuple[int, int]]:
        """Walk to root; union incremental edge sets (Theorem 4)."""
        node = self._resolve_node(k, delta)
        if node is None:
            return set()
        out: set[tuple[int, int]] = set()
        cur: DcNode | None = node
        while cur is not None:
            out |= cur.incremental_edges
            cur = cur.parent
        return out

    def _resolve_node(self, k: int, delta: int) -> DcNode | None:
        candidates = [key for key in self.lookup if key[0] == k and key[1] <= delta]
        if not candidates:
            return None
        return self.lookup[max(candidates, key=lambda x: x[1])]


def dc_index_demo() -> dict[str, Any]:
    from ltx_trainer.kd_truss.running_example import FIG1_TIMESTAMPS
    from ltx_trainer.kd_truss.tc_index import TcIndex

    edges = {tuple(e) for e in FIG1_TIMESTAMPS}
    dc = DcIndex.build(edges, k_max=5, delta_max=6)
    tc = TcIndex.build(edges, k_max=5)
    online = online_kd_truss_query(edges, k=4, delta=1)
    dc_res = dc.dc_query(k=4, delta=1)
    tc_res = tc.tc_query(k=4, delta=1)
    return {
        "lookup_cells": len(dc.lookup),
        "dc_matches_online": dc_res == online,
        "tc_matches_online": tc_res == online,
    }
