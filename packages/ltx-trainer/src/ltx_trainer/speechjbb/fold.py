"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "speechjbb": {
            "task": "lalm_code_switched_audio_jailbreak",
            "paper": "arXiv:2606.06037",
            "judge_labels": ("Refused", "Deflected", "Jailbroken"),
        },
    }
