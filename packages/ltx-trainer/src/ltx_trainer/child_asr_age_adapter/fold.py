"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "child_asr_age_adapter": {
            "task": "children_asr",
            "paper": "arXiv:2606.05440",
            "challenge": "On Top of Pasketti Word Track",
        },
    }
