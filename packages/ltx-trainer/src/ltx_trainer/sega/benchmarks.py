"""Paper table anchors for SEGA."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"SEGA": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "Up to 36 MP on multiple DiT backbones",
        "website": "https://rajabi2001.github.io/sega/",
    }
