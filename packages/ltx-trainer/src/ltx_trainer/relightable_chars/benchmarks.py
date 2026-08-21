"""Paper table anchors for Relightable Holoported Characters."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"Relightable Holoported Characters": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "Lightstage random env + tracking frames",
        "website": "https://vcai.mpi-inf.mpg.de/projects/RHC/",
    }
