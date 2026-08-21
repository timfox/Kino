"""Halo exchange strategies (Sections 3.3, 7.6–7.7)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.stencil import ghost_depth, throughput_mpoints_per_sec

# Table 13 peer vs host-staged (800³ anchor)
TABLE_13_PEER: dict[int, dict[str, tuple[float, float, float]]] = {
    320: {"host": (4.720, 444.33, 1.00), "peer": (1.708, 1227.87, 2.76)},
    480: {"host": (9.628, 735.17, 1.00), "peer": (3.669, 1929.27, 2.62)},
    544: {"host": (12.223, 842.97, 1.00), "peer": (4.791, 2150.36, 2.55)},
    640: {"host": (16.853, 995.48, 1.00), "peer": (6.781, 2474.16, 2.49)},
    800: {"host": (27.278, 1201.24, 1.00), "peer": (11.100, 2951.97, 2.46)},
}

# Table 12 enlarged ghost (800³ row as representative)
TABLE_12_ENLARGED: dict[int, list[tuple[int, float, float, float]]] = {
    800: [
        (1, 11.100, 2951.97, 1.00),
        (2, 10.738, 3051.65, 1.03),
        (4, 10.444, 3137.44, 1.06),
        (8, 10.799, 3034.24, 1.03),
    ],
}


def exchange_strategy_card(method: str) -> dict[str, Any]:
    if method == "peer":
        return {
            "method": "direct_gpu_to_gpu_peer",
            "backend": "CuPy CUDA peer copies",
            "finding": "dominant optimization: 2.46–2.76× vs host-staged",
        }
    if method == "host_staged":
        return {
            "method": "host_staged",
            "path": "GPU→CPU→GPU",
            "finding": "bottleneck for multi-GPU stencil halos",
        }
    return {"method": method}


def compare_exchange(grid_n: int) -> dict[str, Any]:
    row = TABLE_13_PEER.get(grid_n, TABLE_13_PEER[800])
    return {
        "grid": f"{grid_n}³",
        "host_staged": {"runtime_s": row["host"][0], "mpoints_s": row["host"][1]},
        "peer": {"runtime_s": row["peer"][0], "mpoints_s": row["peer"][1]},
        "speedup": row["peer"][2],
    }


def enlarged_ghost_sweep(grid_n: int) -> dict[str, Any]:
    rows = TABLE_12_ENLARGED.get(grid_n, TABLE_12_ENLARGED[800])
    return {
        "grid": f"{grid_n}³",
        "intervals": [
            {
                "s": s,
                "ghost_depth": ghost_depth(s),
                "runtime_s": rt,
                "mpoints_s": mp,
                "speedup_vs_s1": sp,
            }
            for s, rt, mp, sp in rows
        ],
        "best_s": 4,
        "note": "s=8 slower than s=4 due to redundant ghost work",
    }


def estimate_runtime_stub(
    nx: int,
    ny: int,
    nz: int,
    n_steps: int,
    *,
    n_gpus: int = 4,
    exchange: str = "peer",
    comm_interval: int = 1,
) -> float:
    """Analytic stub scaled from paper anchors (not a full FDTD solve)."""
    n = nx
    base = {320: 1.708, 480: 3.669, 544: 4.791, 640: 6.781, 800: 11.100}.get(n, 11.100 * (n / 800) ** 3)
    scale = (100 / max(n_steps, 1))
    if exchange == "host_staged":
        peer = TABLE_13_PEER.get(n, TABLE_13_PEER[800])["peer"]
        base = peer[0] * peer[2]
    if comm_interval > 1:
        row = {r[0]: r[3] for r in TABLE_12_ENLARGED.get(n, TABLE_12_ENLARGED[800])}
        base /= row.get(comm_interval, 1.0)
    if n_gpus < 4:
        base *= 1.2
    return base * scale
