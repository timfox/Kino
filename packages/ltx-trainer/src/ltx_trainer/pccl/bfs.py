"""BFS pathfinding over TEN (Algorithm 2)."""

from __future__ import annotations

from ltx_trainer.pccl.conditions import CollectiveCondition
from ltx_trainer.pccl.ten import TimeExpandedNetwork


def bfs_pathfind(ten: TimeExpandedNetwork, condition: CollectiveCondition) -> dict[int, list[tuple[int, int]]]:
    """Return paths mapping each dest in condition.dests to list of (t, next_npu) hops."""
    src = condition.src
    dests = set(condition.dests)
    t = ten.next_available_time(src, 0)
    visited: set[int] = {src}
    # predecessor: npu -> (prev_npu, (t, npu))
    pred: dict[int, tuple[int | None, tuple[int, int] | None]] = {src: (None, None)}

    while not dests.issubset(visited) and t < ten.max_timesteps:
        frontier = list(visited)
        for current in frontier:
            for nxt in ten.next_devices(current, t):
                if nxt in visited:
                    continue
                visited.add(nxt)
                pred[nxt] = (current, (t, nxt))
        t += 1

    paths: dict[int, list[tuple[int, int]]] = {}
    for dest in condition.dests:
        if dest not in pred:
            paths[dest] = []
            continue
        hops: list[tuple[int, int]] = []
        node = dest
        chain: list[tuple[int, int]] = []
        while pred[node][1] is not None:
            chain.append(pred[node][1])  # type: ignore[arg-type]
            node = pred[node][0]  # type: ignore[assignment]
        hops = list(reversed(chain))
        paths[dest] = hops
    return paths


def paths_to_algorithm(
    condition: CollectiveCondition,
    paths: dict[int, list[tuple[int, int]]],
) -> dict[str, Any]:
    """Human-readable synthesis result for one condition."""
    return {
        "chunk_id": condition.chunk_id,
        "src": condition.src,
        "dests": sorted(condition.dests),
        "paths": {str(k): v for k, v in paths.items()},
    }
