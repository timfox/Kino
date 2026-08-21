"""Adversarial Flow Distillation for AR video (Luo et al., arXiv:2605.26105)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AFDConfig:
    """Defaults from paper Sec. 3–4 and Table 3 (Appendix 7)."""

    video_height: int = 480
    video_width: int = 832
    student_backbones: tuple[str, ...] = field(
        default_factory=lambda: ("self_forcing", "causal_forcing")
    )
    teacher_api: str = "seedance_2.0"

    # LoRA / discriminator
    lora_rank: int = 256
    lora_alpha: int = 256
    discriminator_lr: float = 5e-6

    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    ema_decay: float = 0.99
    precision: str = "bfloat16"

    # DiffusionNFT
    nft_beta: float = 0.1
    prior_weight: float = 1e-4
    advantage_clip_max: float = 5.0

    # Rollout / adaptation
    adaptation_samples: int = 200
    batch_size: int = 32
