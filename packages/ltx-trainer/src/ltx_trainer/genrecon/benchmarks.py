"""Paper table anchors for GenRecon."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"GenRecon": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "~16% vs cutting-edge reconstruction baselines",
        "website": "https://kasothaphie.github.io/GenRecon/",
    }
