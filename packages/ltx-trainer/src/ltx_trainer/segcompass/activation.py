"""SAE activation pattern stubs (Fig. 4–5)."""

from __future__ import annotations

from typing import Literal

TokenType = Literal["question", "cot", "instance", "background"]


def avg_active_features(
    token_type: TokenType,
    threshold: float,
    *,
    backbone: str = "qwen2.5-vl-7b",
) -> float:
    """Toy activation counts vs threshold — monotonic with type salience."""
    base = {
        "qwen2.5-vl-7b": {
            "instance": 280.0,
            "cot": 250.0,
            "question": 120.0,
            "background": 40.0,
        },
        "llava-1.5-7b": {
            "instance": 35.0,
            "cot": 30.0,
            "question": 15.0,
            "background": 5.0,
        },
    }[backbone][token_type]
    scale = max(0.1, 1.0 - threshold)
    return base * scale


def instance_coverage_topk(top_k_pct: float, *, dataset: str = "grefcoco") -> float:
    """Fig. 5 — coverage rate vs random baseline (toy)."""
    random_baseline = top_k_pct
    lift = {"refcocog": 0.55, "grefcoco": 0.60, "reasonseg": 0.58}.get(dataset, 0.55)
    return min(1.0, random_baseline + lift * (1.0 - top_k_pct))
