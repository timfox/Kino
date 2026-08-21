"""Collective algorithm synthesis (Algorithm 3)."""

from __future__ import annotations

import time
from typing import Any

from ltx_trainer.pccl.bfs import bfs_pathfind, paths_to_algorithm
from ltx_trainer.pccl.conditions import (
    CollectiveCondition,
    assign_condition_distances,
    conditions_all_gather,
    conditions_all_to_all,
    conditions_all_to_allv,
)
from ltx_trainer.pccl.config import PcclConfig
from ltx_trainer.pccl.ten import TimeExpandedNetwork
from ltx_trainer.pccl.topology import hypercube_3d, mesh_2d, ring_topology, shortest_path_unweighted


def build_adjacency(cfg: PcclConfig) -> dict[int, list[int]]:
    if cfg.topology == "ring":
        return ring_topology(cfg.n_npus)
    if cfg.topology == "hypercube":
        side = max(2, round(cfg.n_npus ** (1 / 3)))
        return hypercube_3d(side)
    w = cfg.mesh_width or int(cfg.n_npus**0.5)
    return mesh_2d(w, w)


def conditions_for_collective(cfg: PcclConfig, group: list[int]) -> list[CollectiveCondition]:
    coll = cfg.collective.lower()
    if coll == "all_to_all":
        return conditions_all_to_all(group)
    if coll == "all_to_allv":
        return conditions_all_to_allv(group, send_counts={group[0]: 2} if group else None)
    return conditions_all_gather(group)


def synthesize(cfg: PcclConfig | None = None) -> dict[str, Any]:
    """Run PCCL BFS synthesis for a process group on target topology."""
    cfg = cfg or PcclConfig()
    adj = build_adjacency(cfg)
    n = len(adj)
    group = cfg.process_group or list(range(1, n + 1))
    ten = TimeExpandedNetwork.from_adjacency(adj, max_timesteps=cfg.max_timesteps)

    sp = lambda s, d: shortest_path_unweighted(adj, s, d)
    conditions = conditions_for_collective(cfg, group)
    assign_condition_distances(conditions, sp)
    conditions.sort(key=lambda c: c.dist, reverse=True)

    t0 = time.perf_counter()
    algorithm: list[dict[str, Any]] = []
    for c in conditions:
        paths = bfs_pathfind(ten, c)
        algorithm.append(paths_to_algorithm(c, paths))
        for hops in paths.values():
            prev = c.src
            for t, dest in hops:
                ten.disabled.add((t, prev, dest))
                prev = dest

    elapsed_ms = (time.perf_counter() - t0) * 1000
    makespan = _estimate_makespan(algorithm)

    return {
        "collective": cfg.collective,
        "topology": cfg.topology,
        "process_group": group,
        "n_conditions": len(conditions),
        "algorithm": algorithm[: min(8, len(algorithm))],
        "algorithm_truncated": len(algorithm) > 8,
        "synthesis_time_ms": round(elapsed_ms, 3),
        "estimated_makespan_steps": makespan,
        "ten": ten.snapshot(),
        "process_group_aware": True,
    }


def _estimate_makespan(algorithm: list[dict[str, Any]]) -> int:
    max_t = 0
    for entry in algorithm:
        for hops in entry.get("paths", {}).values():
            if hops:
                max_t = max(max_t, max(t for t, _ in hops))
    return max_t + 1


def reduce_from_broadcast(broadcast_algo: list[dict[str, Any]]) -> dict[str, str]:
    """Section 4.5: reverse broadcast + reduction for Reduce."""
    return {
        "method": "reverse_broadcast_with_reduction",
        "source_collective": "broadcast",
        "note": "Reduce-Scatter ← reverse All-Gather; All-Reduce = RS + AG",
    }
