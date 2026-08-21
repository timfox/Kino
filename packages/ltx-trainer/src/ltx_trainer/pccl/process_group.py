"""Process group awareness (Sections 2.2, 6.4)."""

from __future__ import annotations

from typing import Any


def overlay_process_groups(
    cluster_npus: int,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe multiple concurrent process groups (Fig 15 style)."""
    return {
        "cluster_npus": cluster_npus,
        "groups": groups,
        "pccl_behavior": [
            "synthesize per-group conditions independently",
            "BFS may route via NPUs outside process group",
            "avoid subgraph-only synthesis that wastes links",
        ],
    }


def speedup_vs_direct(
    *,
    n_process_groups: int,
    mesh_width: int = 8,
) -> dict[str, Any]:
    """Stub Fig 16/19: PCCL vs CCL Direct bandwidth scaling."""
    base = 2.68
    # benefit decreases as concurrent groups increase (Fig 19)
    factor = max(1.0, base - 0.15 * max(0, n_process_groups - 1))
    return {
        "n_process_groups": n_process_groups,
        "mesh_width": mesh_width,
        "normalized_bandwidth_pccl": round(factor, 2),
        "normalized_bandwidth_ccl_direct": 1.0,
        "speedup": round(factor, 2),
    }
