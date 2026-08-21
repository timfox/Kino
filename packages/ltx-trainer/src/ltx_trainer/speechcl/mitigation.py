"""Mitigation mechanisms — §3."""

from __future__ import annotations

from enum import Enum
from typing import Any


class MitigationMechanism(str, Enum):
    REPLAY = "replay"
    REGULARIZATION = "regularization"
    ARCHITECTURAL_ISOLATION = "architectural_isolation"


def describe_mitigation(mech: MitigationMechanism) -> dict[str, Any]:
    cards: dict[MitigationMechanism, dict[str, Any]] = {
        MitigationMechanism.REPLAY: {
            "idea": "Revisit previous data distributions to anchor geometry",
            "speech_examples": [
                "raw waveform / feature replay (ASR, TTS, audio classification)",
                "text-speech instruction mixing (LALM Stage 2→3)",
                "SFT data replay (Stage 3→4)",
            ],
            "limits": ["privacy of biometric voice data", "storage at foundation scale"],
        },
        MitigationMechanism.REGULARIZATION: {
            "idea": "Soft constraints on updates without old data",
            "speech_examples": [
                "EWC / LwF on early ASR",
                "cross-modal distillation (preference alignment)",
            ],
            "limits": [
                "parameter-level constraints insufficient for entangled speech embeddings",
            ],
        },
        MitigationMechanism.ARCHITECTURAL_ISOLATION: {
            "idea": "Freeze backbone; update lightweight modules",
            "speech_examples": [
                "adapters",
                "LoRA on Whisper for multilingual ASR",
                "frozen text LLM during speech encoder alignment",
            ],
            "limits": [
                "entanglement not aligned with parameter modularity in audio encoders",
            ],
        },
    }
    return {"mechanism": mech.value, **cards[mech]}


def mitigation_reference_map() -> list[dict[str, Any]]:
    """Curated speech-CL method → mechanism mapping (§3, paper refs)."""
    return [
        {"method": "Lifelong E2E ASR", "mechanism": "replay", "ref": "Chang et al., 2021"},
        {"method": "Multilingual TTS CL", "mechanism": "replay", "ref": "Yang et al., 2021"},
        {"method": "Environmental sound CL", "mechanism": "replay", "ref": "Xiao et al., 2022b"},
        {"method": "EWC / LwF ASR", "mechanism": "regularization", "ref": "Ghorbani et al., 2019"},
        {"method": "Multi-dialect acoustic models", "mechanism": "regularization", "ref": "Houston & Kirchhoff, 2020"},
        {"method": "Adapter incremental AST", "mechanism": "architectural_isolation", "ref": "Selvaraj et al., 2023"},
        {"method": "LoRA multilingual Whisper", "mechanism": "architectural_isolation", "ref": "Xu et al., 2024"},
        {"method": "PACE pretrained audio CL", "mechanism": "regularization", "ref": "Li et al., 2026"},
    ]
