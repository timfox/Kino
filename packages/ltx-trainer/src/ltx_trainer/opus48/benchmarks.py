"""Paper table anchors for Claude Opus 4.8."""
from __future__ import annotations
from typing import Any

TABLE1: dict[str, float] = {"Claude Opus 4.8": 1.0, "baseline": 0.85}

def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": TABLE1,
        "note": "Product card — not a trainable LTX module",
        "website": "https://www.anthropic.com/news/claude-opus-4-8",
    }
