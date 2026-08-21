"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "costa": {
            "task": "ad_detection_cs_cond_tts_augmentation",
            "paper": "arXiv:2606.06170",
            "dataset": "ADReSS",
        },
    }
