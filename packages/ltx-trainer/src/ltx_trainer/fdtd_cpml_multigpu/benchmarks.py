"""Paper tables and positioning (Tables 1–14, Figures)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.constants import (
    BEST_ENLARGED_GHOST_S,
    CPML_OVERHEAD_MAX_PCT,
    PEER_SPEEDUP_RANGE,
    SINGLE_GPU_MPOINTS_RANGE,
)


def table_i_positioning() -> list[dict[str, bool | str]]:
    return [
        {"study": "Classical FDTD/PML [1,10,12]", "multi_gpu": False, "cpml": True, "decomp_compare": False, "peer_exchange": False, "enlarged_ghost": False},
        {"study": "GPU stencil/temporal blocking [2,5,13]", "multi_gpu": True, "cpml": False, "decomp_compare": False, "peer_exchange": False, "enlarged_ghost": True},
        {"study": "GPU 3D FDTD [6,7]", "multi_gpu": True, "cpml": False, "decomp_compare": False, "peer_exchange": False, "enlarged_ghost": False},
        {"study": "CPML FDTD [10,11]", "multi_gpu": False, "cpml": True, "decomp_compare": False, "peer_exchange": False, "enlarged_ghost": False},
        {"study": "Present work", "multi_gpu": True, "cpml": True, "decomp_compare": True, "peer_exchange": True, "enlarged_ghost": True},
    ]


def table_10_single_gpu() -> list[dict[str, Any]]:
    return [
        {"grid": "320³", "cpml": False, "runtime_s": 0.635, "mpoints_s": 3304.48},
        {"grid": "320³", "cpml": True, "runtime_s": 0.638, "mpoints_s": 3289.58, "overhead_pct": 0.45},
        {"grid": "800³", "cpml": False, "runtime_s": 11.204, "mpoints_s": 2924.73},
        {"grid": "800³", "cpml": True, "runtime_s": 11.221, "mpoints_s": 2920.36, "overhead_pct": 0.15},
    ]


def table_11_strong_scaling_rtx8000() -> list[dict[str, Any]]:
    return [
        {"grid": "320³", "gpus": 1, "runtime_s": 0.897, "speedup": 1.00, "efficiency_pct": 100.0},
        {"grid": "320³", "gpus": 2, "runtime_s": 0.864, "speedup": 1.04, "efficiency_pct": 51.9},
        {"grid": "800³", "gpus": 2, "runtime_s": 9.658, "speedup": 1.51, "efficiency_pct": 75.7},
        {"grid": "800³", "gpus": 4, "runtime_s": 10.458, "speedup": 1.40, "efficiency_pct": 34.9},
        {"grid": "1024³", "gpus": 1, "runtime_s": None, "note": "OOM"},
        {"grid": "1024³", "gpus": 4, "runtime_s": 18.528, "note": "memory-capacity regime"},
    ]


def table_8_reflection_ratio() -> list[dict[str, Any]]:
    return [
        {"cpml_cells": 0, "ratio": 9.438e-1},
        {"cpml_cells": 8, "ratio": 8.049e-2},
        {"cpml_cells": 12, "ratio": 6.714e-3},
        {"cpml_cells": 16, "ratio": 5.581e-4},
        {"cpml_cells": 24, "ratio": 3.328e-5},
    ]


def summary_anchors() -> dict[str, Any]:
    return {
        "single_gpu_mpoints_s_range": SINGLE_GPU_MPOINTS_RANGE,
        "cpml_overhead_below_pct": CPML_OVERHEAD_MAX_PCT,
        "peer_exchange_speedup_range": PEER_SPEEDUP_RANGE,
        "best_enlarged_ghost_s": BEST_ENLARGED_GHOST_S,
        "best_decomposition": "pencil_yz",
        "hardware_rtx6000": "4× Quadro RTX 6000, optimus node, PCIe/NUMA peer access",
        "hardware_rtx8000": "Quadro RTX 8000 strong-scaling / OOM 1024³",
    }
