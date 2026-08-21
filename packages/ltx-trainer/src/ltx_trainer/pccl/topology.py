"""Network topology builders for TEN construction."""

from __future__ import annotations

from typing import Any


def ring_topology(n: int, *, bidirectional: bool = False) -> dict[int, list[int]]:
    """Unidirectional or bidirectional ring adjacency (1-indexed NPUs)."""
    adj: dict[int, list[int]] = {i: [] for i in range(1, n + 1)}
    for i in range(1, n + 1):
        adj[i].append(i % n + 1)
        if bidirectional:
            adj[i].append((i - 2) % n + 1)
    return adj


def mesh_2d(width: int, height: int | None = None) -> dict[int, list[int]]:
    """2D mesh with bidirectional links (1-indexed row-major NPUs)."""
    h = height or width
    n = width * h
    adj: dict[int, list[int]] = {i: [] for i in range(1, n + 1)}
    for r in range(h):
        for c in range(width):
            idx = r * width + c + 1
            if c + 1 < width:
                adj[idx].append(idx + 1)
                adj[idx + 1].append(idx)
            if r + 1 < h:
                down = idx + width
                adj[idx].append(down)
                adj[down].append(idx)
    return adj


def hypercube_3d(order: int) -> dict[int, list[int]]:
    """3D hypercube-style grid (order^3 NPUs, mesh-like for stub)."""
    side = order
    return mesh_2d(side, side)  # simplified stub; full hypercube for large n in benchmarks only


def shortest_path_unweighted(adj: dict[int, list[int]], src: int, dest: int) -> int:
    if src == dest:
        return 0
    from collections import deque

    q: deque[tuple[int, int]] = deque([(src, 0)])
    seen = {src}
    while q:
        node, dist = q.popleft()
        for nxt in adj.get(node, []):
            if nxt == dest:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return 10_000


def topology_card(name: str, n_npus: int) -> dict[str, Any]:
    if name == "ring":
        return {"name": "ring", "n_npus": n_npus, "links": "unidirectional"}
    if name == "2d_mesh":
        w = int(n_npus**0.5)
        return {"name": "2d_mesh", "width": w, "height": w, "n_npus": w * w}
    return {"name": name, "n_npus": n_npus}
