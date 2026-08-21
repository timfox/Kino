"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "figma": {
            "task": "fine_grained_music_retrieval",
            "paper": "arXiv:2606.06615",
            "dataset": "FGMCaps",
        },
    }
