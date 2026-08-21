"""Three pressing challenges (Fig. 1a)."""

from __future__ import annotations

from typing import Any


def three_challenges() -> list[dict[str, Any]]:
    return [
        {
            "id": "affective_data_scarcity",
            "name": "Affective Data Scarcity",
            "summary": (
                "Subjective labels, costly multi-annotator protocols, and fragmented "
                "emotion theories limit scale of high-quality emotional data."
            ),
            "remedies": ("affective_data_augmentation",),
        },
        {
            "id": "multimodal_affective_gap",
            "name": "Multimodal Affective Gap",
            "summary": (
                "Intra-modality misalignment between factual and emotional features; "
                "inter-modality heterogeneity and conflicting cues across I/V/A/T."
            ),
            "remedies": ("multimodal_affective_representation",),
        },
        {
            "id": "affective_interpretation_opacity",
            "name": "Affective Interpretation Opacity",
            "summary": (
                "Traditional MER emphasizes classification over explanation; LLMs enable "
                "step-by-step affective reasoning but raise hallucination and consistency risks."
            ),
            "remedies": ("multimodal_affective_reasoning",),
        },
    ]
