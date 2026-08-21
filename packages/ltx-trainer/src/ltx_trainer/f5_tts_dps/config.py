"""F5-TTS-DPS WildSpoof 2026 TTS track (arXiv:2605.23859)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class F5TtsDpsConfig:
    paper_arxiv: str = "arXiv:2605.23859"
    challenge: str = "WildSpoof 2026 TTS Track"
    base_model: str = "SWivid/F5-TTS F5TTS_v1_Base"
    lalm_audio_scorer: str = "Qwen2.5-Omni"
    llm_text_scorer: str = "Qwen3-30B-A3B"
    ema_beta: float = 0.99
    train_epochs: int = 10
    learning_rate: float = 1e-6
    audio_score_min: float = 7.0
    dev_utmos: float = 3.20
    dev_wer: float = 8.65
    dev_spk_sim: float = 0.508
    dev_sds: float = 0.108
    test_adcf_t01: float = 0.1582
    test_adcf_t02: float = 0.5233
    test_adcf_t08: float = 0.2562
