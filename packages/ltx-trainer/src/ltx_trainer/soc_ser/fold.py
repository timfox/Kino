"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "soc_ser": {
            "task": "speech_emotion_recognition",
            "paper": "arXiv:2606.06550",
            "aggregation": "second_order_correlation_lem",
        },
    }
