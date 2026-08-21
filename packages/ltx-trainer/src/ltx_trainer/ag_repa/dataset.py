"""Training data card (LibriSpeech + AudioSet, paper Sec. 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ag_repa.config import AgRepaConfig


def dataset_card(cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    return {
        "unified_training": True,
        "speech": c.speech_dataset,
        "general_audio": c.audio_dataset,
        "token_topologies": {
            "config_a": c.token_config_a,
            "config_b": c.token_config_b,
        },
        "teachers": {
            "semantic": c.semantic_teacher,
            "acoustic": c.acoustic_teacher,
        },
        "stage1": "Qwen3-0.6B AR LLM for acoustic token prediction",
        "stage2": "DiT Flow Matching mel synthesis + Vocos vocoder",
        "fad_metric": "VGGish @ 16 kHz",
        "checkpoint_steps": c.train_checkpoint_steps,
        "warmup_probe_steps": c.warmup_probe_steps,
    }
