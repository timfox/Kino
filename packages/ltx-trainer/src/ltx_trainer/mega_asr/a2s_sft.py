"""Acoustic-to-Semantic Progressive SFT schedule (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def a2s_sft_phases() -> list[dict[str, Any]]:
    """Three-phase curriculum: encoder-aligner → LLM → joint."""
    return [
        {
            "phase": "I",
            "name": "encoder_aligner_acoustic",
            "wer_curriculum": ["<30%", "<50%", "<70%"],
            "trainable": ["audio_encoder", "aligner"],
            "lr_encoder": 1e-3,
            "lr_llm": None,
        },
        {
            "phase": "II",
            "name": "llm_semantic",
            "wer_curriculum": ["full targeted split"],
            "trainable": ["llm"],
            "lr_encoder": None,
            "lr_llm": 2e-5,
        },
        {
            "phase": "III",
            "name": "joint_acoustic_semantic",
            "wer_curriculum": ["full targeted split"],
            "trainable": ["audio_encoder", "aligner", "llm"],
            "lr_encoder": 2e-6,
            "lr_llm": 2e-6,
        },
    ]


def filter_learnability(wer: float, *, max_wer: float = 0.70) -> bool:
    """Discard samples with WER > 70% for training stability (Sec. 3.2)."""
    return wer <= max_wer
