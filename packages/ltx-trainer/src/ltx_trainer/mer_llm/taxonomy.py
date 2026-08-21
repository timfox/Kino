"""MER-with-LLMs taxonomy (Fig. 2) and five sub-tasks (Fig. 3)."""

from __future__ import annotations

from typing import Any


def five_subtasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "GVEC",
            "name": "Generic Visual Emotion Comprehension",
            "modalities": ("I", "V", "V+A"),
            "goal": "Viewer-side emotion elicited by visual stimuli",
            "example_datasets": ("EmoSet", "Emotion6", "VECBench"),
        },
        {
            "id": "VTSA",
            "name": "Visual-Textual Sentiment Analysis",
            "modalities": ("I", "T"),
            "goal": "Emotion in image–text social posts",
            "example_datasets": ("MVSA-S", "MVSA-M", "MM-BigBench"),
        },
        {
            "id": "SEC",
            "name": "Speech Emotion Comprehension",
            "modalities": ("A",),
            "goal": "Emotion in speech via classification or captioning",
            "example_datasets": ("MELD", "IEMOCAP", "EMOSEC"),
        },
        {
            "id": "FER",
            "name": "Facial Expression Recognition",
            "modalities": ("I", "V"),
            "goal": "Emotional state from face-centric video/image",
            "example_datasets": ("RAF-DB", "AffectNet", "DFEW"),
        },
        {
            "id": "CMER",
            "name": "Conversation-based Multimodal Emotion Recognition",
            "modalities": ("V", "A", "T"),
            "goal": "Speaker emotion in audiovisual dialogue",
            "example_datasets": ("MOSI", "MOSEI", "MER2024", "OV-MERD"),
        },
    ]


def taxonomy_branches() -> list[dict[str, Any]]:
    """Three branches aligned with Fig. 2."""
    return [
        {
            "id": "affective_data_augmentation",
            "section": "§3",
            "name": "Affective Data Augmentation",
            "children": (
                {
                    "name": "Training-free Sample Configuration",
                    "examples": ("SoV", "SoVTP", "NarraCap", "ICL (Wu et al., 2025a)"),
                },
                {
                    "name": "Emotion Data Annotation",
                    "sub_branches": (
                        "Dataset Engineering (EmoVIT, VEC-CoT, MER-Caption, …)",
                        "Benchmark Construction (VECBench, OV-MERD, EmotionHallucer, …)",
                    ),
                },
            ),
        },
        {
            "id": "multimodal_affective_representation",
            "section": "§4",
            "name": "Multimodal Affective Representation",
            "children": (
                {
                    "name": "Perceptual Emotion Mapping",
                    "examples": ("SEPM", "EmoDETective", "BLSP-Emo", "EmoChat", "Facial-R1"),
                },
                {
                    "name": "Multimodal Emotion Coordination",
                    "examples": ("EmoVIT", "AffectGPT", "M3F", "Omni-Emotion", "EMO-LLaMA"),
                },
            ),
        },
        {
            "id": "multimodal_affective_reasoning",
            "section": "§5",
            "name": "Multimodal Affective Reasoning",
            "children": (
                {
                    "name": "Emotion Explanation and Hallucination",
                    "examples": ("Facial-R1", "AlignCap", "PEP-MEK", "ERV", "MulCoT-RD"),
                },
                {
                    "name": "Subjective Emotion Reasoning",
                    "examples": ("EmoCaliber", "Agent-MER", "AffectGPT-R1"),
                },
            ),
        },
    ]


def future_directions() -> tuple[str, ...]:
    """Sec. 7 open directions."""
    return (
        "Unified and Generalized MER across GVEC/VTSA/SEC/FER/CMER",
        "Mechanism-level exploration (which parameters drive affect perception)",
        "Subjectivity-embraced frameworks (open-vocabulary, confidence verbalization)",
        "Agentic emotion understanding (observation, planning, tools, real-time feedback)",
        "Safety, bias, and cultural adaptation in deployment",
    )
