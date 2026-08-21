"""Minimum time span (Def. 1) and δ-triangle lists (Def. 9)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from ltx_trainer.kd_truss.running_example import get_timestamps, normalize_edge


def minimum_time_span(
    u: int,
    v: int,
    w: int,
    *,
    timestamps: dict[tuple[int, int], tuple[int, ...]] | None = None,
) -> int:
    """mts(∆) = min max(|t1-t2|, |t2-t3|, |t3-t1|) over timestamp triples."""

    def _ts(a: int, b: int) -> tuple[int, ...]:
        if timestamps is not None:
            return timestamps[normalize_edge(a, b)]
        return get_timestamps(a, b)

    ta = _ts(u, v)
    tb = _ts(v, w)
    tc = _ts(w, u)
    best = 10**9
    for t1 in ta:
        for t2 in tb:
            for t3 in tc:
                span = max(abs(t1 - t2), abs(t2 - t3), abs(t3 - t1))
                if span < best:
                    best = span
    return best


def enumerate_triangles(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    verts = sorted(adj)
    tris: list[tuple[int, int, int]] = []
    for u in verts:
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w <= v:
                    continue
                tris.append((u, v, w))
    return tris


def delta_triangle_list(
    edges: Iterable[tuple[int, int]],
    *,
    timestamps: dict[tuple[int, int], tuple[int, ...]] | None = None,
) -> tuple[dict[tuple[int, int, int], int], list[list[tuple[int, int, int]]]]:
    """Return triangle mts map and S^∆_δ buckets (Def. 9)."""
    mts_map: dict[tuple[int, int, int], int] = {}
    delta_max = 0
    for tri in enumerate_triangles(edges):
        mts = minimum_time_span(*tri, timestamps=timestamps)
        mts_map[tri] = mts
        delta_max = max(delta_max, mts)
    buckets: list[list[tuple[int, int, int]]] = [[] for _ in range(delta_max + 1)]
    for tri, mts in mts_map.items():
        buckets[mts].append(tri)
    return mts_map, buckets
