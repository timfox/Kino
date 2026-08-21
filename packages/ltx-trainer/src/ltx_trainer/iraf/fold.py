"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "iraf": {
            "task": "full_duplex_spoken_dialogue",
            "paper": "arXiv:2606.06559",
            "module": "interference_resilient_adaptive_fusion",
        },
    }
