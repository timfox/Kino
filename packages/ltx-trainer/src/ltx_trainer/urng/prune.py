"""UG unified pruning and candidate generation (Alg. 1–3)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ltx_trainer.urng.intervals import FLAG_IF, FLAG_IS, Interval, phi_if, phi_is


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def generate_candidates(
    node_ids: list[str],
    vectors: dict[str, tuple[float, ...]],
    intervals: dict[str, Interval],
    *,
    ef_spatial: int,
    ef_attribute: int,
) -> dict[str, set[str]]:
    """Alg. 1 stub: spatial kNN + interval-sorted neighbors."""
    per_side = max(1, ef_attribute // 8)
    c_spa: dict[str, set[str]] = {u: set() for u in node_ids}
    c_attr: dict[str, set[str]] = {u: set() for u in node_ids}

    for u in node_ids:
        ranked = sorted(node_ids, key=lambda v: _dist(vectors[u], vectors[v]))
        c_spa[u] = {v for v in ranked[1 : ef_spatial + 1]}

    keys = {
        "l": lambda i: intervals[i].left,
        "r": lambda i: intervals[i].right,
        "mid": lambda i: intervals[i].mid,
        "len": lambda i: intervals[i].length,
    }
    for key_fn in keys.values():
        order = sorted(node_ids, key=key_fn)
        pos = {nid: idx for idx, nid in enumerate(order)}
        for u in node_ids:
            i = pos[u]
            lo = max(0, i - per_side)
            hi = min(len(order), i + per_side + 1)
            for j in range(lo, hi):
                if j != i:
                    c_attr[u].add(order[j])

    out: dict[str, set[str]] = {}
    for u in node_ids:
        out[u] = (c_spa[u] | c_attr[u]) - {u}
    return out


def unified_prune(
    u: str,
    candidates: Iterable[str],
    vectors: dict[str, tuple[float, ...]],
    intervals: dict[str, Interval],
    *,
    max_edges_if: int,
    max_edges_is: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, str]]]:
    """Alg. 3: return retained (neighbor, mask) and repair pairs ΔW."""
    i_u = intervals[u]
    vec_u = vectors[u]
    sorted_c = sorted(set(candidates) - {u}, key=lambda v: _dist(vec_u, vectors[v]))
    neighbors: list[tuple[str, int]] = []
    repair: list[tuple[str, str]] = []
    cnt_if = 0
    cnt_is = 0

    for v in sorted_c:
        i_v = intervals[v]
        mask = FLAG_IF | FLAG_IS
        if i_u.intersect(i_v) is None:
            mask &= ~FLAG_IS

        for w, w_mask in neighbors:
            if mask == 0:
                break
            if _dist(vectors[v], vectors[w]) >= _dist(vec_u, vectors[v]):
                continue
            i_w = intervals[w]
            if mask & FLAG_IF and w_mask & FLAG_IF and phi_if(i_u, i_v, i_w):
                mask &= ~FLAG_IF
                repair.append((w, v))
            if mask & FLAG_IS and w_mask & FLAG_IS and phi_is(i_u, i_v, i_w):
                mask &= ~FLAG_IS
                repair.append((w, v))

        if mask & FLAG_IF:
            if cnt_if < max_edges_if:
                cnt_if += 1
            else:
                mask &= ~FLAG_IF
        if mask & FLAG_IS:
            if cnt_is < max_edges_is:
                cnt_is += 1
            else:
                mask &= ~FLAG_IS

        if mask:
            neighbors.append((v, mask))

    return neighbors, repair


def build_ug(
    node_ids: list[str],
    vectors: dict[str, tuple[float, ...]],
    intervals: dict[str, Interval],
    *,
    ef_spatial: int = 128,
    ef_attribute: int = 300,
    max_edges_if: int = 256,
    max_edges_is: int = 256,
    iterations: int = 5,
) -> dict[str, list[tuple[str, int]]]:
    """Alg. 2: iterative UG construction."""
    c0 = generate_candidates(
        node_ids,
        vectors,
        intervals,
        ef_spatial=min(ef_spatial, len(node_ids)),
        ef_attribute=ef_attribute,
    )
    repair: dict[str, set[str]] = {u: set() for u in node_ids}
    c_cur = c0

    final: dict[str, list[tuple[str, int]]] = {u: [] for u in node_ids}
    for _ in range(max(1, iterations)):
        repair_next: dict[str, set[str]] = {u: set() for u in node_ids}
        for u in node_ids:
            pool = set(c_cur[u]) | repair[u]
            nbrs, delta = unified_prune(
                u,
                pool,
                vectors,
                intervals,
                max_edges_if=max_edges_if,
                max_edges_is=max_edges_is,
            )
            final[u] = nbrs
            c_cur[u] = {v for v, _ in nbrs}
            for w, v in delta:
                repair_next[w].add(v)
        repair = repair_next

    # symmetrize for undirected search
    adj: dict[str, dict[str, int]] = {u: {} for u in node_ids}
    for u, nbrs in final.items():
        for v, mask in nbrs:
            prev = adj[u].get(v, 0)
            adj[u][v] = prev | mask
            prev_r = adj[v].get(u, 0)
            adj[v][u] = prev_r | mask

    return {u: sorted(adj[u].items()) for u in node_ids}


def edge_mask(
    graph: dict[str, list[tuple[str, int]]],
    a: str,
    b: str,
) -> int:
    for v, m in graph.get(a, []):
        if v == b:
            return m
    return 0
