"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "usad2": {
            "task": "universal_audio_encoder",
            "paper": "arXiv:2606.06444",
            "frontend": "usad_2.0_plus",
        },
    }
