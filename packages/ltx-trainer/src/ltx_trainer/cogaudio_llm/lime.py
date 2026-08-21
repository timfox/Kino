"""LIME-440K dataset statistics and semantic decoupling stub (§2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cogaudio_llm.config import CogAudioLlmConfig


def lime_statistics() -> list[dict[str, Any]]:
    """Table 1 — LIME-440K subset breakdown."""
    return [
        {
            "subset": "LIME-Core Part A (CN)",
            "lang": "CN",
            "emo_int": "7×3",
            "speakers": "~200",
            "hours": 263.9,
            "utterances": 223_884,
        },
        {
            "subset": "LIME-Core Part B (EN)",
            "lang": "EN",
            "emo_int": "7×3",
            "speakers": "~200",
            "hours": 113.8,
            "utterances": 96_000,
        },
        {
            "subset": "LIME-Aug Part C (ECD-TSE)",
            "lang": "EN",
            "emo_int": "5×1",
            "speakers": 12,
            "hours": 90.3,
            "utterances": 84_000,
        },
        {
            "subset": "LIME-Aug Part D (ESD)",
            "lang": "Mix",
            "emo_int": "5×1",
            "speakers": 20,
            "hours": 29.1,
            "utterances": 35_000,
        },
    ]


def decouple_text_emotions(text: str, emotions: list[str]) -> dict[str, Any]:
    """One-text multi-emotion decoupling stub."""
    return {
        "text": text,
        "emotions": emotions,
        "lexically_identical": True,
        "requires_paralinguistics": len(set(emotions)) >= 2,
    }


def lime_demo(*, cfg: CogAudioLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CogAudioLlmConfig()
    stats = lime_statistics()
    total_utts = sum(s["utterances"] for s in stats)
    example = decouple_text_emotions(
        "I never imagined the project would end like this.",
        ["Happy", "Sad", "Angry"],
    )
    return {
        "total_utterances": total_utts,
        "total_hours": cfg.lime_hours,
        "subsets": len(stats),
        "decoupling_example": example,
        "annotation_acceptance_pct": 93.0,
    }
