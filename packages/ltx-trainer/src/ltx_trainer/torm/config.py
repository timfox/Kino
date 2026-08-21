"""TORM internalized spatial-temporal video reasoning (Liang et al., arXiv:2605.26014)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TORMConfig:
    """Defaults from paper Sec. 3–4 and experiment setup."""

    backbone: str = "Qwen2.5-VL-7B-Instruct"
    num_latent_slots: int = 8
    hidden_dim: int = 3584
    latent_align_weight: float = 0.1  # λ in Eq. (4)

    latent_start_token: str = "<|latent_start|>"
    latent_pad_token: str = "<|latent_pad|>"
    latent_end_token: str = "<|latent_end|>"

    max_frames: int = 32
    thought_video_generator: str = "Wan-2.1"
    keyframe_selector: str = "GPT-4o"
    thought_videos_train: int = 5000

    # Optimization (Sec. 4.2)
    learning_rate: float = 1e-5
    batch_size: int = 1
    epochs_per_stage: int = 1
