"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "mvse_amd": {
            "task": "query_adaptive_av_person_retrieval",
            "paper": "arXiv:2606.05931",
            "presence_types": ("AVP", "AoP", "VoP"),
        },
    }
