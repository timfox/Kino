"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "xvec_emo": {
            "task": "xvector_emotion_arithmetic_lm_tts",
            "paper": "arXiv:2606.05367",
            "operand": "ecapa_xvector",
            "delta_eecs_en": 0.288,
        },
    }
