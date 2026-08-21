"""Paper table anchors for CubePart."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"CubePart": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "Roblox/cube two-stage global shape + part decode",
        "website": "https://cubepart.github.io/",
    }
