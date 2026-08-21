"""Paper table anchors for Gamma-World."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"Gamma-World": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "2→4 player zero-shot without retrain",
        "website": "https://research.nvidia.com/labs/sil/projects/gamma-world/",
    }
