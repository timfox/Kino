"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "aac_audioset": {
            "task": "automated_audio_captioning",
            "paper": "arXiv:2606.05717",
            "datasets": ("Clotho-V2", "AudioCaps"),
        },
    }
