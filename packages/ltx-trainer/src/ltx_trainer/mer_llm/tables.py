"""Paper Tables 1–5 excerpts (MER-with-LLMs survey)."""

from __future__ import annotations

from typing import Any


def table1_emotion_datasets_excerpt() -> list[dict[str, Any]]:
    """Table 1 subset — training / benchmark rows with distinguishing focus."""
    return [
        {
            "dataset": "EmoVIT",
            "modality": "I,T",
            "sub_task": "GVEC",
            "samples": 51200,
            "focus": "First emotion-centric visual instruction tuning",
        },
        {
            "dataset": "VEC-CoT",
            "modality": "I,T",
            "sub_task": "GVEC",
            "samples": 143446,
            "focus": "Structured affective reasoning at scale",
        },
        {
            "dataset": "MER-Caption",
            "modality": "V,A,T",
            "sub_task": "CMER",
            "samples": 115595,
            "focus": "Large-scale CMER instruction data",
        },
        {
            "dataset": "OV-MERD",
            "modality": "V,A,T",
            "sub_task": "CMER",
            "samples": 332,
            "labels": "OV",
            "focus": "Open-vocabulary multi-label emotions",
        },
        {
            "dataset": "EmotionHallucer",
            "modality": "I,V,A,T",
            "sub_task": "CMER",
            "samples": 2742,
            "focus": "Emotion reasoning hallucination benchmark",
        },
    ]


def table2_perceptual_mapping_excerpt() -> list[dict[str, Any]]:
    """Table 2 subset — perceptual emotion mapping methods."""
    return [
        {"method": "SEPM", "sub_task": "GVEC", "technique": "Zero-Shot", "encoder": "Sharpened"},
        {"method": "EmoVIT", "sub_task": "GVEC", "technique": "SFT", "fusion": "Q-former"},
        {"method": "BLSP-Emo", "sub_task": "SEC", "technique": "SFT", "design": "Unimodal before multimodal"},
        {"method": "Facial-R1", "sub_task": "FER", "technique": "SFT+RL", "focus": "Explanation consistency"},
        {"method": "EmoChat", "sub_task": "GVEC,VTSA,FER,CMER", "technique": "SFT", "focus": "Feature alignment"},
    ]


def table5_quantitative_excerpt() -> list[dict[str, Any]]:
    """Table 5 — representative MER-with-LLMs results (paper-reported)."""
    return [
        {
            "sub_task": "GVEC",
            "method": "EmoVIT",
            "branch": "Multimodal Emotion Coordination",
            "dataset": "EmoSet",
            "metric": "Acc",
            "value_pct": 83.36,
        },
        {
            "sub_task": "VTSA",
            "method": "MulCoT-RD",
            "branch": "Emotion Explanation and Hallucination",
            "dataset": "MVSA-M",
            "metric": "Acc",
            "value_pct": 77.20,
        },
        {
            "sub_task": "FER",
            "method": "Facial-R1",
            "branch": "Emotion Explanation and Hallucination",
            "dataset": "RAF-DB",
            "metric": "Acc",
            "value_pct": 92.10,
        },
        {
            "sub_task": "CMER",
            "method": "AffectGPT-R1",
            "branch": "Subjective Emotion Reasoning",
            "dataset": "OV-MERD+",
            "metric": "WAF",
            "value_pct": 68.39,
        },
        {
            "sub_task": "SEC",
            "method": "AlignCap",
            "branch": "Emotion Explanation and Hallucination",
            "dataset": "EMOSEC",
            "metric": "BLEU@4",
            "value": 9.8,
        },
    ]


def fig1b_paradigm_progress() -> list[dict[str, Any]]:
    """Fig. 1(b) — small model SOTA vs zero-shot MLLM vs MER-with-LLMs (accuracy %)."""
    return [
        {
            "sub_task": "GVEC",
            "dataset": "EmoSet",
            "small_scale_sota": 73.5,
            "general_mllm_zero_shot": 66.8,
            "mer_with_llms": 83.4,
            "leader_method": "EmoVIT",
        },
        {
            "sub_task": "VTSA",
            "dataset": "MVSA-M",
            "small_scale_sota": 63.8,
            "general_mllm_zero_shot": 48.9,
            "mer_with_llms": 72.7,
            "leader_method": "EmoChat",
        },
        {
            "sub_task": "FER",
            "dataset": "AffectNet",
            "small_scale_sota": 86.3,
            "general_mllm_zero_shot": 82.8,
            "mer_with_llms": 88.5,
        },
        {
            "sub_task": "SEC",
            "dataset": "MELD",
            "small_scale_sota": 54.8,
            "general_mllm_zero_shot": 32.9,
            "mer_with_llms": 57.3,
            "leader_method": "BLSP-Emo",
        },
    ]
