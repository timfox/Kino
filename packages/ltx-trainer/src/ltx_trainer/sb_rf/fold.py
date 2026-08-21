"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sb_rf": {
            "task": "one_step_speech_enhancement",
            "paper": "arXiv:2606.05575",
            "nfe": 1,
            "tracks": ["VB-DMD", "low-SNR"],
        },
    }
