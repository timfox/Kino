"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "mdd_lssg": {
            "task": "mispronunciation_detection_diagnosis",
            "paper": "arXiv:2606.05569",
            "dataset": "L2-ARCTIC",
            "detection_f1": 0.5952,
        },
    }
