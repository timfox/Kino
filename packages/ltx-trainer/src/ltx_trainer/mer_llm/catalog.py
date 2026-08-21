"""Representative methods index (Fig. 2 leaves) for agent lookup."""

from __future__ import annotations

from typing import Any


def representative_methods() -> list[dict[str, Any]]:
    """Curated leaf methods from the survey taxonomy."""
    return [
        {"method": "EmoVIT", "branch": "Multimodal Emotion Coordination", "sub_tasks": ("GVEC",)},
        {"method": "AffectGPT", "branch": "Multimodal Emotion Coordination", "sub_tasks": ("CMER",)},
        {"method": "AffectGPT-R1", "branch": "Subjective Emotion Reasoning", "sub_tasks": ("CMER",)},
        {"method": "EmoChat", "branch": "Perceptual Emotion Mapping", "sub_tasks": ("GVEC", "VTSA", "FER", "CMER")},
        {"method": "Facial-R1", "branch": "Emotion Explanation and Hallucination", "sub_tasks": ("FER",)},
        {"method": "BLSP-Emo", "branch": "Perceptual Emotion Mapping", "sub_tasks": ("SEC",)},
        {"method": "EmoCaliber", "branch": "Subjective Emotion Reasoning", "sub_tasks": ("GVEC",)},
        {"method": "Agent-MER", "branch": "Subjective Emotion Reasoning", "sub_tasks": ("CMER",)},
        {"method": "OV-MERD", "branch": "Benchmark Construction", "sub_tasks": ("CMER",), "type": "dataset"},
        {"method": "EmotionHallucer", "branch": "Benchmark Construction", "sub_tasks": ("CMER",), "type": "dataset"},
    ]


def methods_for_subtask(subtask_id: str) -> list[dict[str, Any]]:
    sid = subtask_id.upper()
    return [m for m in representative_methods() if sid in m.get("sub_tasks", ())]


def research_radar_brief(*, vertical: str = "game_live_ops") -> dict[str, Any]:
    """One-screen brief for Gopex Research Radar SKU (documents/GOPEX_GDC_2026_STRATEGY.md)."""
    hooks = {
        "game_live_ops": (
            "CMER on dialogue clips supports player-support tone QA; pair with human escalation "
            "(Helpshift pattern). Do not auto-ban from MER labels."
        ),
        "trailer_marketing": (
            "GVEC/VTSA stubs help brief whether key art evokes intended affect before LTX generation; "
            "use LongAV-Compass for minute-scale AV quality, not emotion taxonomy."
        ),
        "political_speech": (
            "Use pathos_mm for rhetorical pathos; MER-with-LLMs for discrete emotion categories "
            "and explainable CoT on multimodal political clips."
        ),
    }
    return {
        "paper": "arXiv:2605.21239",
        "vertical": vertical,
        "so_what": hooks.get(vertical, hooks["game_live_ops"]),
        "try_first": ["mer_llm_framework_card", "mer_llm_benchmarks"],
        "top_methods": [m["method"] for m in representative_methods()[:6]],
        "open_risks": (
            "Zero-shot MLLM gap vs MER-with-LLMs SFT; emotion hallucination; "
            "cultural bias — see future_directions in framework_card."
        ),
    }
