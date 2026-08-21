"""Paper table anchors for MiniCPM5-1B."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"MiniCPM5-1B": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "Hub: openbmb/MiniCPM5-1B",
        "website": "https://huggingface.co/openbmb/MiniCPM5-1B",
    }
