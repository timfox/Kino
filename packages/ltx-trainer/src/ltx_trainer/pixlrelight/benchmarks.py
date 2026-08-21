"""Paper table anchors for PIXLRelight."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"PIXLRelight": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "<0.1s per image feed-forward",
        "website": "https://mlfarinha.github.io/pixl-relight/",
    }
