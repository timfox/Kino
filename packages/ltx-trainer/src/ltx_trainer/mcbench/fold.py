"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "mcbench": {
            "task": "multicontext_omni_safety",
            "paper": "arXiv:2606.05177",
            "modalities": ["vision", "audio", "speech"],
            "scenarios": 1196,
        },
    }
