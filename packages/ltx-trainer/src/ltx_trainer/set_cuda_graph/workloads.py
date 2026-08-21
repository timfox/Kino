"""Six evaluation workloads (§5.1, Fig. 4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.constants import WORKLOADS


def workload_catalog() -> list[dict[str, Any]]:
    notes = {
        "Sobel": "normalization, edge detect, mean filter, threshold, blend",
        "GEMM": "tile-based dense matrix multiply",
        "BP": "single-layer training step with on-device synthetic minibatch",
        "KNN": "brute-force k-NN classification",
        "Hotspot": "iterative thermal simulation (memory-bound, up to 90% DRAM BW)",
        "SSSP": "Bellman–Ford frontier relaxation",
    }
    return [{**dict(w), "description": notes.get(w["name"], "")} for w in WORKLOADS]


def workload_by_name(name: str) -> dict[str, Any]:
    for w in workload_catalog():
        if w["name"].lower() == name.lower():
            return w
    raise KeyError(f"unknown workload: {name}")
