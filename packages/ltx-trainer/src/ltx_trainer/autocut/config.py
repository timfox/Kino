"""AutoCut configuration (arXiv:2603.28366)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RQVAEConfig:
    """Residual-quantized VAE settings (Supp. Table 3–4)."""

    input_dim: int = 128
    quant_heads: int = 8
    codebook_size: int = 256
    codebook_dim: int = 128
    encoder_mlp: tuple[int, ...] = (512, 512)
    decoder_mlp: tuple[int, ...] = (512, 512)
    loss_type: str = "cosine"
    target_cosine_video: float = 0.89
    target_cosine_audio: float = 0.96


@dataclass
class AutoCutConfig:
    """End-to-end ad video editing via multimodal discretization."""

    paper_arxiv: str = "2603.28366"
    github_url: str = "https://github.com/AdAutoCut/Autocut"
    base_llm: str = "Qwen3-8B"
    visual_encoder: str = "ResNet-50 (CNNv2, contrastive pretrain)"
    audio_encoder: str = "PANNs (AudioSet)"
    alignment_samples: int = 700_000
    sft_samples: int = 100_000
    eval_videos: int = 364
    low_fps: int = 1
    high_fps: int = 20
    video_feature_dim: int = 128
    audio_feature_dim: int = 2048
    video_rqvae: RQVAEConfig = field(default_factory=lambda: RQVAEConfig(input_dim=128))
    audio_rqvae: RQVAEConfig = field(
        default_factory=lambda: RQVAEConfig(
            input_dim=2048,
            codebook_dim=256,
            encoder_mlp=(1024, 512),
            decoder_mlp=(512, 1024),
        )
    )
    alignment_cutoff: int = 8096
    sft_cutoff: int = 4000
    alignment_lr: float = 5e-5
    sft_lr: float = 1e-5
    alignment_epochs: int = 5
    sft_epochs: int = 3
    inference_cost_per_100_videos_usd: float = 0.015
    gpt4o_cost_per_100_videos_usd: float = 2.5
    random_seed: int = 42

    @classmethod
    def production(cls) -> AutoCutConfig:
        return cls()
