"""WavTTS raw waveform zero-shot TTS (arXiv:2606.03455)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WavTTSConfig:
    paper_arxiv: str = "arXiv:2606.03455"
    title: str = "WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling"
    github: str = "https://github.com/cwx-worst-one/WavTTS"
    project_page: str = "https://wavtts.github.io"

    sample_rate_hz: int = 16_000
    patch_size: int = 160
    patch_rate_hz: int = 100
    params_m: float = 673.0
    training_data: str = "Emilia (~100K hrs)"

    # Training (Section 4)
    lambda_mel: float = 0.05
    scale_k: float = 9.0
    logit_normal_mu: float = -0.8
    logit_normal_sigma: float = 0.8
    t_clip_max: float = 0.98
    cfg_drop_prob: float = 0.1

    # Inference
    nfe: int = 50
    cfg_scale: float = 3.0
    polyshift_p: float = 2.0
    polyshift_s: float = 3.0

    # Table 1 — Seed-TTS (WavTTS)
    seed_en_wer: float = 1.50
    seed_en_sim: float = 0.65
    seed_en_utmos: float = 3.92
    seed_zh_cer: float = 1.59
    seed_zh_sim: float = 0.73
    seed_zh_utmos: float = 3.08

    # Table 2 — supervised TTS
    ljspeech_wer: float = 3.43
    ljspeech_utmos: float = 4.39
    librispeech_wer: float = 2.02

    demo_batch: int = 2
    demo_time_samples: int = 1600
