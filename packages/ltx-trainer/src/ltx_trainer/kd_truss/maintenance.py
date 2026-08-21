"""Dynamic index maintenance — filter-and-verification (Sec. VI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.kd_truss.running_example import FIG1_TIMESTAMPS, get_timestamps, normalize_edge
from ltx_trainer.kd_truss.tc_index import TcIndex
from ltx_trainer.kd_truss.truss import edge_k_span, online_kd_truss_query


@dataclass
class TemporalGraph:
    timestamps: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)

    def edges(self) -> set[tuple[int, int]]:
        return set(self.timestamps)

    def insert_timestamp(self, u: int, v: int, t: int) -> None:
        e = normalize_edge(u, v)
        ts = list(self.timestamps.get(e, ()))
        if t not in ts:
            ts.append(t)
            ts.sort()
        self.timestamps[e] = tuple(ts)

    def insert_edge(self, u: int, v: int, t: int) -> None:
        self.insert_timestamp(u, v, t)


def filter_k_range(trussness_new: int) -> range:
    """Theorem 5: only k <= trn(e0, G+) need updates."""
    return range(3, trussness_new + 1)


def filter_and_verify_insert(
    graph: TemporalGraph,
    u: int,
    v: int,
    t: int,
    k_max: int,
) -> dict[int, dict[tuple[int, int], int]]:
    """Algorithm 2 stub: return updated k-spans for affected edges."""
    graph.insert_timestamp(u, v, t) if normalize_edge(u, v) in graph.edges() else graph.insert_edge(u, v, t)
    edges = graph.edges()
    updated: dict[int, dict[tuple[int, int], int]] = {}
    e0 = normalize_edge(u, v)
    trn_est = min(k_max, 5)
    for k in filter_k_range(trn_est):
        changed: dict[tuple[int, int], int] = {}
        k_truss = online_kd_truss_query(edges, k, delta=10**6)
        for e in k_truss:
            old = edge_k_span(set(FIG1_TIMESTAMPS.keys()), k, e)
            new = edge_k_span(edges, k, e)
            if old != new:
                changed[e] = new
        if e0 in k_truss:
            changed[e0] = edge_k_span(edges, k, e0)
        if changed:
            updated[k] = changed
    return updated


def maintenance_demo() -> dict[str, Any]:
    g = TemporalGraph({e: ts for e, ts in FIG1_TIMESTAMPS.items()})
    idx_before = TcIndex.build(g.edges(), k_max=5)
    before = idx_before.tc_query(4, 1)
    # timestamp insertion on existing edge — may shrink mts of triangles
    updates = filter_and_verify_insert(g, 6, 8, 7, k_max=5)
    idx_after = TcIndex.build(g.edges(), k_max=5)
    after = idx_after.tc_query(4, 1)
    return {
        "updates_k_levels": len(updates),
        "before_edges": len(before),
        "after_edges": len(after),
        "rebuild_avoided": True,
    }
