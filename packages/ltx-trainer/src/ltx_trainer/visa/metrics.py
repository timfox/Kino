"""MMAR accuracy and rubrics helpers."""

from __future__ import annotations

from typing import Any


def rubrics_components(
    *,
    factuality: float,
    coherence: float,
    completeness: float,
) -> dict[str, float]:
    """Instance-level rubric sub-scores (stub weights)."""
    weights = {"factuality": 0.4, "coherence": 0.35, "completeness": 0.25}
    score = (
        factuality * weights["factuality"]
        + coherence * weights["coherence"]
        + completeness * weights["completeness"]
    )
    return {
        "factuality": factuality,
        "coherence": coherence,
        "completeness": completeness,
        "rubrics_score": round(score * 100, 2),
    }


def mc_accuracy(predicted: str, gold: str) -> float:
    return 1.0 if predicted.strip().upper() == gold.strip().upper() else 0.0


def aggregate_accuracy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return round(100.0 * sum(scores) / len(scores), 2)
