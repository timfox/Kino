"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "univoice": {
            "task": "unified_speech_singing_cfm",
            "paper": "arXiv:2606.05852",
            "modalities": ("speech", "singing"),
        },
    }
