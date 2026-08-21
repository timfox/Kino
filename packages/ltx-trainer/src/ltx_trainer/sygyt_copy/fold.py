"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sygyt_copy": {
            "task": "biphonic_articulatory_copy_synthesis",
            "paper": "arXiv:2606.04943",
            "style": "sygyt",
            "sample_rate_hz": 16000,
        },
    }
