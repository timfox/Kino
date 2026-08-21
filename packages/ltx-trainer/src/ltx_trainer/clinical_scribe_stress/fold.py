"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "clinical_scribe_stress": {
            "task": "paired_acoustic_stress_test",
            "paper": "arXiv:2606.05909",
            "metrics": ("WER", "NegErr", "SCER", "Unsafe"),
        },
    }
