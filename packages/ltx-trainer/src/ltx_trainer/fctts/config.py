"""FC-TTS zero-shot TTS with disentangled style and timbre (arXiv:2605.24618)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FcttsConfig:
    paper_arxiv: str = "arXiv:2605.24618"
    demo_url: str = "https://qualcomm-ai-research.github.io/fc-tts"
    num_params_m: int = 204
    train_iterations: int = 200_000
    batch_size: int = 64
    learning_rate: float = 2e-4
    duration_nfe: int = 8
    mel_nfe: int = 32
    cfg_scale: float = 4.0
    cfg_dropout: float = 0.15
    lambda_cfm: float = 5.0
    lambda_blur: float = 1.0
    lambda_ccl_pro: float = 0.2
    lambda_ccl_spk: float = 0.5
    facodec_prosody_levels: int = 1
    facodec_content_levels: int = 2
    facodec_detail_levels: int = 3
