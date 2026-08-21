"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "ucs_sfx": {
            "task": "sfx_ucs_unification",
            "paper": "arXiv:2606.05571",
            "envsound_clips": 58057,
            "ucs_categories": 59,
        },
    }
