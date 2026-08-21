"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "film_spk_asr": {
            "task": "pathological_asr",
            "paper": "arXiv:2606.06211",
            "corpora": ("NeuroVoz", "TORGO"),
        },
    }
