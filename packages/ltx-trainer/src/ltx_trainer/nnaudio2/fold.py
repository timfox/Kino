"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "nnaudio2": {
            "task": "differentiable_audio_frontends",
            "paper": "arXiv:2606.05394",
            "istft_freq_scale": "no",
            "icqt_snr_db_min": 30.0,
        },
    }
