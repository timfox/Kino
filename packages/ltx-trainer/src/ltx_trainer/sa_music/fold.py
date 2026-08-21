"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sa_music": {
            "task": "south_asian_music_llm_benchmark",
            "paper": "arXiv:2606.05522",
            "benchmark_questions": 504,
            "gemini_style_accuracy": 0.40,
        },
    }
