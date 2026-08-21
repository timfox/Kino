"""Paper table anchors for Bonsai Image 4B."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"Bonsai Image 4B": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "0.93 GB 1-bit / 1.21 GB ternary transformer",
        "website": "https://prismml.com/news/bonsai-image-4b",
    }
