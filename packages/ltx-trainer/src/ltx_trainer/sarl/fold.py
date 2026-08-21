"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sarl": {
            "task": "spatial_audio_representation_probing",
            "paper": "arXiv:2606.05544",
            "probe_tasks": 7,
            "encoders": 13,
        },
    }
