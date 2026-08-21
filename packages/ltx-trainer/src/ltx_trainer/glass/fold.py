"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "glass": {
            "task": "grpo_lora_style_steering",
            "paper": "arXiv:2606.05889",
            "axes": ("speed", "pitch"),
        },
    }
