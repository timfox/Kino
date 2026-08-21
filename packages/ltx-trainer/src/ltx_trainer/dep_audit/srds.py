"""SRDS topic-density slice stub — §2.5 Probe D."""

from __future__ import annotations

from typing import Any


def topic_score_rubric() -> dict[str, str]:
    return {
        "0": "no depression-related topic",
        "1": "weak / ambiguous cue",
        "2": "clear depression-related topic",
        "3": "strong, direct, repeated self-relevant cue",
    }


def classify_chunk_stub(text: str) -> dict[str, Any]:
    """Toy annotator: keyword density proxy for topic_score."""
    keywords = ("sleep", "mood", "sad", "hopeless", "anhedonia", "suicid")
    hits = sum(1 for k in keywords if k in text.lower())
    score = min(3, hits)
    return {
        "topic_score": score,
        "self_relevance": "self" if hits else "generic",
        "confidence": 0.7 + 0.05 * hits,
    }


def heavy_neutral_delta(
    heavy_scores: list[float],
    neutral_scores: list[float],
) -> dict[str, float]:
    if len(heavy_scores) != len(neutral_scores):
        raise ValueError("paired participant lists must match")
    gaps = [h - n for h, n in zip(heavy_scores, neutral_scores, strict=True)]
    return {
        "mean_shift": float(sum(gaps) / len(gaps)),
        "positive_fraction": sum(1 for g in gaps if g > 0) / len(gaps),
    }
