"""Online (k, δ)-truss query via truss peeling (Sec. III)."""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Iterable

from ltx_trainer.kd_truss.mts import delta_triangle_list, enumerate_triangles, minimum_time_span
from ltx_trainer.kd_truss.running_example import normalize_edge


def _triangles_for_edge(
    tri_mts: dict[tuple[int, int, int], int],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    by_edge: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for tri in tri_mts:
        u, v, w = tri
        for e in ((u, v), (v, w), (w, u)):
            by_edge[normalize_edge(*e)].append(tri)
    return by_edge


def delta_support(
    edge: tuple[int, int],
    tri_mts: dict[tuple[int, int, int], int],
    delta: int,
    active_edges: set[tuple[int, int]] | None = None,
) -> int:
    e = normalize_edge(*edge)
    if active_edges is not None and e not in active_edges:
        return 0
    tris = [t for t, m in tri_mts.items() if e[0] in t and e[1] in t and m <= delta]
    if active_edges is not None:
        tris = [
            t
            for t in tris
            if all(normalize_edge(t[i], t[(i + 1) % 3]) in active_edges for i in range(3))
        ]
    return len(tris)


def online_kd_truss_query(
    edges: Iterable[tuple[int, int]],
    k: int,
    delta: int,
) -> set[tuple[int, int]]:
    """Index-free Online-Query: peel edges with δ-support < k-2."""
    edge_set = {normalize_edge(u, v) for u, v in edges}
    tri_mts, _ = delta_triangle_list(edge_set)
    active = set(edge_set)
    tri_by_edge = _triangles_for_edge(tri_mts)

    sup: dict[tuple[int, int], int] = {
        e: delta_support(e, tri_mts, delta, active) for e in active
    }
    heap: list[tuple[int, tuple[int, int]]] = [(sup[e], e) for e in active]
    heapq.heapify(heap)
    threshold = k - 2

    while heap:
        s, e = heapq.heappop(heap)
        if e not in active or sup[e] != s:
            continue
        if s >= threshold:
            break
        active.remove(e)
        for tri in tri_by_edge.get(e, []):
            if tri_mts[tri] > delta:
                continue
            for pair in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                e2 = normalize_edge(*pair)
                if e2 in active:
                    sup[e2] -= 1
                    heapq.heappush(heap, (sup[e2], e2))
    return active


def edge_k_span(
    edges: Iterable[tuple[int, int]],
    k: int,
    edge: tuple[int, int],
    *,
    delta_max: int = 64,
) -> int:
    """k-span: minimum δ with e ∈ (k, δ)-truss (Def. 5)."""
    e = normalize_edge(*edge)
    edge_set = {normalize_edge(u, v) for u, v in edges}
    for delta in range(0, delta_max + 1):
        if e in online_kd_truss_query(edge_set, k, delta):
            return delta
    return delta_max


def static_k_truss(edges: Iterable[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    """Classical k-truss (δ = ∞): all triangles count regardless of mts."""
    edge_set = {normalize_edge(u, v) for u, v in edges}
    tris = enumerate_triangles(edge_set)
    tri_by_edge = _triangles_for_edge({t: 0 for t in tris})
    active = set(edge_set)
    sup = {e: len(tri_by_edge.get(e, [])) for e in active}
    heap = [(sup[e], e) for e in active]
    heapq.heapify(heap)
    threshold = k - 2
    while heap:
        s, e = heapq.heappop(heap)
        if e not in active or sup[e] != s:
            continue
        if s >= threshold:
            break
        active.remove(e)
        for tri in tri_by_edge.get(e, []):
            for pair in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                e2 = normalize_edge(*pair)
                if e2 in active:
                    sup[e2] -= 1
                    heapq.heappush(heap, (sup[e2], e2))
    return active


def truss_demo() -> dict[str, object]:
    from ltx_trainer.kd_truss.running_example import EXAMPLE2_TIMESTAMPS, EXAMPLE5_CORE, FIG1_TIMESTAMPS

    edges = FIG1_TIMESTAMPS.keys()
    t46 = online_kd_truss_query(edges, k=4, delta=6)
    t41 = online_kd_truss_query(edges, k=4, delta=1)
    ex2_mts = {
        (2, 3, 8): minimum_time_span(2, 3, 8, timestamps=EXAMPLE2_TIMESTAMPS),
        (2, 7, 8): minimum_time_span(2, 7, 8, timestamps=EXAMPLE2_TIMESTAMPS),
    }
    sup_28 = delta_support((2, 8), ex2_mts, 6)
    return {
        "k4_delta6_edges": len(t46),
        "k4_delta1_edges": len(t41),
        "example5_core_in_truss": EXAMPLE5_CORE <= set(t41),
        "mts_v2_v3_v8": ex2_mts[(2, 3, 8)],
        "mts_v2_v7_v8": ex2_mts[(2, 7, 8)],
        "delta_sup_v2_v8_at_6": sup_28,
    }
