"""Paper table anchors for SCOPE."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"SCOPE": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "69K clips × 7 games, zero-shot 2→4 players",
        "website": "https://z2tong.github.io/SCOPE/",
    }
