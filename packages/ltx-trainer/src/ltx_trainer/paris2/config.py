"""Paris 2.0 decentralized video DDM (Rouzbayani et al., arXiv:2605.26064)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paris2Config:
    """Stage 1 three-expert setup (Sec. 3–4)."""

    num_experts: int = 3
    expert_params_b: float = 11.0
    router_params_m: float = 100.0

    latent_channels: int = 16
    latent_frames: int = 8
    latent_height: int = 32
    latent_width: int = 32

    clip_dim: int = 768
    dino_dim: int = 1024
    router_hidden: int = 512
    top_k_experts: int = 1

    vae: str = "HunyuanVAE"
    text_encoders: tuple[str, ...] = field(
        default_factory=lambda: ("T5-v1.1-XXL", "CLIP-ViT-L/14")
    )
    expert_backbone: str = "FLUX-style MM-DiT (OpenSora recipe)"
    init_weights: str = "FLUX.1-dev"

    # Generation protocol (Sec. 4.1)
    eval_resolution: int = 256
    eval_num_clips: int = 2048
    sampler: str = "Euler-50"
    cfg_scale: float = 7.5

    # Training
    flow_weight: float = 1.0
    router_weight: float = 1.0
    learning_rate: float = 1e-4
