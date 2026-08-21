"""LoRA-Key configuration (arXiv:2605.29569)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRAKeyConfig:
    paper_arxiv: str = "arXiv:2605.29569"
    message_bits: int = 48
    lora_rank: int = 64
    watermark_lora_rank: int = 64
    alpha_style: float = 1.0
    gamma_watermark: float = 1.0
    # Stage 1 latent prior (Eq. 9)
    lambda_mse: float = 1.0
    lambda_lpips: float = 0.1
    # Stage 2 GOP + consistency
    gop_eps: float = 1e-8
    dino_feature_dim: int = 384
    # Verification (Eq. 16–18)
    target_fpr: float = 1e-6
    base_model: str = "Stable Diffusion v1.4"
