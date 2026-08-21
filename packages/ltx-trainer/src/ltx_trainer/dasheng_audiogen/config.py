"""Dasheng AudioGen config (arXiv:2605.27838)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DashengAudioGenConfig:
    paper_arxiv: str = "arXiv:2605.27838"
    demo_url: str = "https://nieeim.github.io/Dasheng-AudioGen-Web/"
    # Architecture
    dit_layers: int = 32
    dit_hidden: int = 1536
    dit_params_b: float = 2.0
    tokenizer_decoder_layers: int = 12
    tokenizer_decoder_params_m: float = 173.0
    text_encoder: str = "Flan-T5-Large"
    text_encoder_params_m: float = 780.0
    latent_dim: int = 1280
    text_dim: int = 256
    latent_hz: float = 25.0
    clip_duration_s: float = 10.0
    fm_steps: int = 25
    cfg_scale: float = 5.0
    field_dropout: float = 0.2
    # Training
    train_hours: float = 77000.0
    train_steps: int = 800_000
    batch_size: int = 256
    lr: float = 5e-4
    # Table 2 anchors (Ours)
    audiocaps_fad: float = 3.19
    musiccaps_fad: float = 1.37
    librispeech_wer_pct: float = 10.77
    librispeech_utmos: float = 3.12
    # Table 3 SMA (hardest mixed)
    mecat_sma_fad: float = 2.17
    mecat_sma_wer_pct: float = 28.98
    expert_pipeline_sma_fad: float = 6.38
    expert_pipeline_sma_wer_pct: float = 62.14
    # Table 4 structured vs unstructured (LibriTTS WER)
    unstructured_wer_pct: float = 52.0
    # PAFI / human (SMA)
    pafi_ours_sma: float = 3.61
    pafi_gt_sma: float = 3.60
    pafi_expert_sma: float = 2.88
