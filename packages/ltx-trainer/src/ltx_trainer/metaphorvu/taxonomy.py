"""Eight-type video metaphor taxonomy (Sec. 2.1)."""

from __future__ import annotations

from typing import Any

METAPHOR_TYPES: tuple[str, ...] = (
    "Body Language",
    "Atmosphere Language",
    "Cultural Symbol",
    "Naturalistic Symbol",
    "Causal Montage",
    "Analogical Montage",
    "Surreal Narrative",
    "Performative Narrative",
)

TYPE_SHORT: dict[str, str] = {
    "Body Language": "Body L.",
    "Atmosphere Language": "Atmosph. L.",
    "Cultural Symbol": "Cultural S.",
    "Naturalistic Symbol": "Natural. S.",
    "Causal Montage": "Causal M.",
    "Analogical Montage": "Analog. M.",
    "Surreal Narrative": "Surreal N.",
    "Performative Narrative": "Perform. N.",
}


def taxonomy_card() -> list[dict[str, str]]:
    """Short descriptions aligned with Figure 2."""
    return [
        {
            "type": "Body Language",
            "channel": "Character body movements and exaggerated actions.",
        },
        {
            "type": "Atmosphere Language",
            "channel": "Color, lighting, and composition mood.",
        },
        {
            "type": "Cultural Symbol",
            "channel": "Cultural artifacts (e.g., Kongming lantern).",
        },
        {
            "type": "Naturalistic Symbol",
            "channel": "Nature/animal symbolism.",
        },
        {
            "type": "Causal Montage",
            "channel": "Cause-effect shot juxtaposition.",
        },
        {
            "type": "Analogical Montage",
            "channel": "Thematically similar shot juxtaposition.",
        },
        {
            "type": "Surreal Narrative",
            "channel": "Cartoon / AI surreal plots.",
        },
        {
            "type": "Performative Narrative",
            "channel": "Human short-drama storytelling.",
        },
    ]


def benchmark_type_stats() -> dict[str, dict[str, Any]]:
    """Table 1 — per-type sample counts and averages."""
    return {
        "Body Language": {"samples": 136, "avg_duration_s": 32.2, "avg_tokens": 111.3},
        "Atmosphere Language": {"samples": 150, "avg_duration_s": 13.1, "avg_tokens": 104.5},
        "Cultural Symbol": {"samples": 62, "avg_duration_s": 23.5, "avg_tokens": 114.4},
        "Naturalistic Symbol": {"samples": 113, "avg_duration_s": 17.3, "avg_tokens": 108.8},
        "Causal Montage": {"samples": 54, "avg_duration_s": 57.7, "avg_tokens": 108.9},
        "Analogical Montage": {"samples": 171, "avg_duration_s": 58.7, "avg_tokens": 124.8},
        "Surreal Narrative": {"samples": 112, "avg_duration_s": 30.4, "avg_tokens": 117.1},
        "Performative Narrative": {"samples": 62, "avg_duration_s": 86.8, "avg_tokens": 118.6},
        "MetaphorVU-Bench": {"samples": 860, "avg_duration_s": 37.2, "avg_tokens": 114.2},
    }
