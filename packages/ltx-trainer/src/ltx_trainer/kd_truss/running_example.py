"""Fig. 1 running example temporal graph (ICDE paper)."""

from __future__ import annotations

from typing import Any

# Edge -> sorted timestamps. Star vertex v5 augments K4 on {6,8,9,10} so each
# clique edge has ≥3 δ-triangles at δ=1 (k=4 peeling stability).
FIG1_TIMESTAMPS: dict[tuple[int, int], tuple[int, ...]] = {
    (5, 6): (8,),
    (5, 8): (8,),
    (5, 9): (8,),
    (5, 10): (8,),
    (6, 8): (8,),
    (6, 9): (8,),
    (6, 10): (8,),
    (8, 9): (8,),
    (8, 10): (8,),
    (9, 10): (8,),
    (2, 3): (2,),
    (3, 8): (2, 4),
    (2, 8): (0, 2, 8),
    (2, 7): (0, 6),
    (3, 7): (3, 5),
    (7, 8): (0, 6),
    (0, 1): (0,),
    (0, 3): (0,),
    (1, 2): (1,),
    (1, 3): (1,),
    (4, 5): (0, 10),
    (4, 6): (0, 10),
    (4, 7): (0, 10),
    (5, 7): (0, 10),
}

# Paper Example 2 (Def. 1–3): e=(2,8); mts({2,3,8})=2, mts({2,7,8})=6.
EXAMPLE2_TIMESTAMPS: dict[tuple[int, int], tuple[int, ...]] = {
    (2, 3): (0,),
    (3, 8): (0,),
    (2, 8): (2,),
    (2, 7): (0,),
    (7, 8): (6,),
}

EXAMPLE5_CORE = frozenset({(6, 8), (6, 9), (6, 10), (8, 9), (8, 10), (9, 10)})


def normalize_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def fig1_temporal_graph() -> dict[str, Any]:
    edges = sorted(FIG1_TIMESTAMPS.keys())
    return {
        "vertices": list(range(11)),
        "edges": edges,
        "timestamps": {f"{u},{v}": list(ts) for (u, v), ts in FIG1_TIMESTAMPS.items()},
        "delta_max": 6,
        "example5_core": sorted(EXAMPLE5_CORE),
    }


def get_timestamps(u: int, v: int) -> tuple[int, ...]:
    return FIG1_TIMESTAMPS[normalize_edge(u, v)]
