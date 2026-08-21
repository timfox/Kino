"""Five evaluated AVSR architectures (§3)."""

from __future__ import annotations

from typing import Any


def model_registry() -> list[dict[str, Any]]:
    return [
        {
            "name": "AV-HuBERT",
            "family": "self_supervised",
            "unified_modality": True,
            "training_labelled_h": 433,
        },
        {
            "name": "Auto-AVSR",
            "family": "supervised_e2e",
            "unified_modality": False,
            "training_labelled_h": 3448,
        },
        {
            "name": "USR",
            "family": "unified_teacher_student",
            "unified_modality": True,
            "training_labelled_h": 433,
        },
        {
            "name": "Whisper-Flamingo",
            "family": "speech_foundation",
            "unified_modality": False,
            "training_labelled_h": 1759,
        },
        {
            "name": "Llama-AVSR",
            "family": "llm_integrated",
            "unified_modality": False,
            "training_labelled_h": 1759,
        },
    ]
