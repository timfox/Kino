"""Paper table anchors for LocateAnything."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"LocateAnything": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "12.7 BPS on H100 vs 1.1 BPS Qwen3-VL",
        "website": "https://research.nvidia.com/labs/lpr/locate-anything/",
    }
