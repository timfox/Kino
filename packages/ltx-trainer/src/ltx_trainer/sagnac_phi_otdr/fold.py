"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "sagnac_phi_otdr": {
            "task": "das_phi_otdr_event_recognition",
            "paper": "arXiv:2606.05754",
            "channels": 12,
            "classes": 6,
        },
    }
