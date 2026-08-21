"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "edr_zs_ser": {
            "task": "zero_shot_cross_lingual_ser",
            "paper": "arXiv:2606.06200",
            "languages": ("EN", "CN", "DE", "FR", "UR"),
        },
    }
