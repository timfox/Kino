"""Multi-GPU domain decomposition layouts (Section 3.3, Table 5)."""

from __future__ import annotations

from typing import Any

# Table 5 baseline (pre-optimized) anchors: (runtime_s, mpoints/s) per grid
TABLE_5_BASELINE: dict[int, dict[str, tuple[float, float]]] = {
    320: {
        "slab_z": (2.357, 889.71),
        "block_xy": (3.417, 613.72),
        "pencil_yz": (2.345, 894.45),
    },
    544: {
        "slab_z": (8.399, 1226.79),
        "block_xy": (10.889, 946.24),
        "pencil_yz": (7.616, 1352.80),
    },
    800: {
        "slab_z": (23.778, 1378.06),
        "block_xy": (29.947, 1094.19),
        "pencil_yz": (21.932, 1494.09),
    },
}


def decomposition_card(name: str) -> dict[str, Any]:
    cards = {
        "slab_z": {"layout": "1×1×4", "split": "z only", "ghost_dirs": ["z"]},
        "block_xy": {"layout": "2×2×1", "split": "x and y", "ghost_dirs": ["x", "y"]},
        "pencil_yz": {"layout": "1×2×2", "split": "y and z; full x local", "ghost_dirs": ["y", "z"]},
    }
    return cards.get(name, {"layout": name})


def select_best_decomposition(grid_n: int) -> dict[str, Any]:
    """Table 5: pencil-yz wins all tested grids on RTX 6000 baseline."""
    row = TABLE_5_BASELINE.get(grid_n, TABLE_5_BASELINE[800])
    best = max(row.items(), key=lambda kv: kv[1][1])
    return {
        "grid": f"{grid_n}³",
        "candidates": {k: {"runtime_s": v[0], "mpoints_s": v[1]} for k, v in row.items()},
        "selected": best[0],
        "throughput_mpoints_s": best[1][1],
        "rationale": "pencil-yz highest Mpoints/s on optimus four-GPU node",
    }


def per_gpu_memory_gib(
    nx: int,
    ny: int,
    nz: int,
    n_gpus: int,
    *,
    decomposition: str = "pencil_yz",
) -> float:
    """Table 14 style: p + 3 velocity + 6 CPML aux fields, float32."""
    fields = 10
    bytes_per_point = 4 * fields
    if decomposition == "pencil_yz" and n_gpus == 4:
        owned = (nx * (ny // 2) * (nz // 2))
    elif n_gpus == 2:
        owned = nx * ny * (nz // 2)
    else:
        owned = nx * ny * nz
    return owned * bytes_per_point / (1024**3)
