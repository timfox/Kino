"""Entry-node acquisition and interval-aware beam search (Alg. 4–5)."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from ltx_trainer.urng.intervals import Interval, QueryType, edge_active, node_valid


@dataclass
class EntryArrays:
    """Sorted-by-left arrays with suffix-min / prefix-max right endpoints."""

    order: list[str]
    left: list[float]
    right: list[float]
    suffix_min_r: list[float]
    prefix_max_r: list[float]


def build_entry_arrays(intervals: dict[str, Interval]) -> EntryArrays:
    order = sorted(intervals, key=lambda k: intervals[k].left)
    left = [intervals[i].left for i in order]
    right = [intervals[i].right for i in order]
    n = len(order)
    suffix_min_r = [0.0] * n
    prefix_max_r = [0.0] * n
    if n:
        suffix_min_r[-1] = right[-1]
        for i in range(n - 2, -1, -1):
            suffix_min_r[i] = min(right[i], suffix_min_r[i + 1])
        prefix_max_r[0] = right[0]
        for i in range(1, n):
            prefix_max_r[i] = max(right[i], prefix_max_r[i - 1])
    return EntryArrays(order, left, right, suffix_min_r, prefix_max_r)


def get_entry_node(arr: EntryArrays, query: Interval, qtype: QueryType) -> str | None:
    """Alg. 5: O(log n) entry node or None if no valid object exists."""
    if qtype in (QueryType.IFANN, QueryType.RFANN):
        lo, hi = 0, len(arr.order)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr.left[mid] < query.left:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(arr.order):
            return None
        if arr.suffix_min_r[lo] <= query.right:
            return arr.order[lo]
        return None

    if qtype in (QueryType.ISANN, QueryType.RSANN):
        lo, hi = 0, len(arr.order)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr.left[mid] <= query.left:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return None
        if arr.prefix_max_r[idx] >= query.right:
            return arr.order[idx]
        return None

    return None


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def interval_aware_beam_search(
    graph: dict[str, list[tuple[str, int]]],
    vectors: dict[str, tuple[float, ...]],
    intervals: dict[str, Interval],
    query_vec: tuple[float, ...],
    query_interval: Interval,
    qtype: QueryType,
    *,
    k: int = 10,
    ef_search: int = 64,
    entry: str | None = None,
) -> list[str]:
    """Alg. 4: greedy beam search on UG under interval predicate."""
    if entry is None:
        arr = build_entry_arrays(intervals)
        entry = get_entry_node(arr, query_interval, qtype)
    if entry is None:
        return []

    visited: set[str] = {entry}
    candidates: list[tuple[float, str]] = []
    results: list[tuple[float, str]] = []

    def push(node: str) -> None:
        d = _dist(vectors[node], query_vec)
        heapq.heappush(candidates, (d, node))
        heapq.heappush(results, (-d, node))
        if len(results) > ef_search:
            heapq.heappop(results)

    if node_valid(intervals[entry], query_interval, qtype):
        push(entry)

    while candidates:
        dist_u, u = heapq.heappop(candidates)
        if results and len(results) >= ef_search:
            worst = -results[0][0]
            if dist_u > worst:
                break
        for v, mask in graph.get(u, []):
            if v in visited:
                continue
            if not edge_active(mask, qtype):
                continue
            if not node_valid(intervals[v], query_interval, qtype):
                continue
            visited.add(v)
            push(v)

    ranked = sorted(results, key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _, nid in ranked:
        if nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        if len(out) >= k:
            break
    return out


def brute_force_query(
    vectors: dict[str, tuple[float, ...]],
    intervals: dict[str, Interval],
    query_vec: tuple[float, ...],
    query_interval: Interval,
    qtype: QueryType,
    *,
    k: int = 10,
) -> list[str]:
    valid = [nid for nid in vectors if node_valid(intervals[nid], query_interval, qtype)]
    valid.sort(key=lambda nid: _dist(vectors[nid], query_vec))
    return valid[:k]
