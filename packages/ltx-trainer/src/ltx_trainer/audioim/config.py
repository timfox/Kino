"""AudioIM — attribute-aware V2A with reference audio (arXiv:2606.07182)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioImConfig:
    paper_arxiv: str = "arXiv:2606.07182"
    title: str = (
        "Audio Imitator: Controlling Timbre and Tempo in Video2Audio Synthesis "
        "with Audio Reference"
    )
    framework: str = "AudioIM"
    backbone: str = "MMAudio L-44.1kHz"
    dataset: str = "VGGSound"
    demo_url: str = "https://anonymousdemo757.github.io/"

    # Architecture (§3.3)
    mm_blocks: int = 7
    sm_blocks: int = 14
    hidden_dim: int = 896
    learning_rate: float = 1e-5
    batch_size: int = 512
    train_steps: int = 800

    # Masking (§2.1): 3s prompt / 5s target within 8s clip
    prompt_seconds: float = 3.0
    target_seconds: float = 5.0
    mask_prompt_ratio: float = 0.375  # 3:5
    mask_target_ratio: float = 0.625
    cfg_dropout: float = 0.10

    # Style encoders (§2.2)
    timbre_encoder: str = "BEATs"
    tempo_encoder: str = "Style Conditioner (RVQ)"
    tempo_codebooks: int = 6

    # Table 1 — AudioIM (ours) anchors
    kl_panns: float = 1.65
    kl_passt: float = 1.44
    ib_score: float = 31.98
    desync: float = 0.49

    # Table 2 — style similarity anchors
    style_kl_panns: float = 1.85
    style_kl_passt: float = 1.63
    ss_mos: float = 4.06
    ss_mos_ci: float = 0.21
