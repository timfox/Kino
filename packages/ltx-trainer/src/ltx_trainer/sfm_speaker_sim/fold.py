"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sfm_speaker_sim": {
            "task": "human_vs_model_speaker_similarity",
            "paper": "arXiv:2606.05739",
            "n_models": 43,
            "metrics": ["LCC", "SRCC", "Frobenius", "spectral"],
        },
    }
