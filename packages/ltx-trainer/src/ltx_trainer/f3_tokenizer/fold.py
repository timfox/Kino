"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "f3_tokenizer": {
            "task": "unified_audio_tokenizer",
            "paper": "arXiv:2606.06357",
            "outputs": ("z_acoustic_latent", "u_representation"),
        },
    }
