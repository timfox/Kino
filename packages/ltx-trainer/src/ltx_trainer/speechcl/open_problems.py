"""Open challenges — §5."""

from __future__ import annotations

from typing import Any


def open_problems() -> list[dict[str, Any]]:
    return [
        {
            "id": "continual_pretraining",
            "title": "Scalable continual pretraining",
            "summary": "Move beyond continual fine-tuning on frozen foundations",
            "ref": "Roth et al., 2024",
        },
        {
            "id": "generative_pseudo_replay",
            "title": "Internal generative pseudo-replay",
            "summary": (
                "Bypass raw biometric voice storage; LALMs self-generate "
                "modality-aligned pseudo-replays from latent space"
            ),
            "ref": "Frascaroli et al., 2024",
        },
        {
            "id": "missing_modality",
            "title": "CL under missing / corrupted modalities",
            "summary": (
                "Dynamic routing or masking to protect cross-modal alignment "
                "when text metadata or audio streams are incomplete"
            ),
            "ref": "Huang et al., 2025",
        },
        {
            "id": "geometry_eval",
            "title": "Representation-geometry evaluation protocols",
            "summary": (
                "Metrics beyond task accuracy: phonetic separability, "
                "speaker manifold structure, paralinguistic organization"
            ),
        },
        {
            "id": "entangled_adaptation",
            "title": "Adapting entangled latent spaces",
            "summary": (
                "PEFT does not isolate geometry when phonetics, speaker, "
                "and paralinguistics share one acoustic manifold"
            ),
            "ref": "Li et al., 2026",
        },
    ]
