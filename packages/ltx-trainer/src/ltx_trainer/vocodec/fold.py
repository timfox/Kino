"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "vocodec": {
            "task": "streamable_neural_speech_codec",
            "paper": "arXiv:2606.05892",
            "target_kbps_libritts": 1.1,
        },
    }
