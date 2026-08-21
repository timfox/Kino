"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "l2_do_entangle": {
            "task": "dual_output_l2_asr",
            "paper": "arXiv:2606.06065",
            "outputs": ("surface", "meaning"),
        },
    }
