"""Wavelet as Tokenizer (WAT) — arXiv:2606.02631."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WATConfig:
    paper_arxiv: str = "arXiv:2606.02631"
    title: str = "Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema"
    author: str = "Shenghao Ding (Yet Another AI)"

    # Datasets (Section 5.1)
    audio_dataset: str = "Speech Commands"
    image_dataset: str = "EuroSAT RGB"
    video_dataset: str = "DAVIS 2017"
    audio_samples: int = 16_384
    image_size: int = 64
    video_frames: int = 8

    # Dense token counts (rate proxy)
    dense_tokens_audio: int = 16_384
    dense_tokens_image: int = 4_096
    dense_tokens_video: int = 32_768

    # Model defaults (Appendix A.2)
    token_width: int = 32
    latent_dim: int = 16
    hidden_dim: int = 64
    audio_scale: float = 4.0
    image_scale: float = 1.0
    video_scale: float = 1.0
    learning_rate: float = 1e-3
    train_steps: int = 300

    # Table 1 — shared schema, audio scale 4
    table1_audio_psnr: float = 39.92
    table1_image_psnr: float = 29.37
    table1_video_psnr: float = 23.93

    # Table 5 — best masked sparse
    masked_video_psnr: float = 34.45
    masked_video_keep: float = 0.50

    # Energy selection gains vs uniform (Table 4 average)
    energy_gain_audio_db: float = 16.73
    energy_gain_image_db: float = 16.90
    energy_gain_video_db: float = 15.86

    demo_batch: int = 2
