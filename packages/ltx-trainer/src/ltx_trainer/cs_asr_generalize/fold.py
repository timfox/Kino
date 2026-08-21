"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "cs_asr_generalize": {
            "task": "code_switching_asr_unseen_pairs",
            "paper": "arXiv:2606.05846",
            "languages": ("EN", "KO", "JA", "DE"),
        },
    }
