"""Configuration for Squeeze-MLLM subject-driven generation (arXiv:2605.26111)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SqueezeMLLMConfig:
    paper_arxiv: str = "arXiv:2605.26111"
    project_page: str = "https://zsh2000.github.io/squeeze-mllm-subject-gen/"

    # Backbone references (paper defaults).
    mllm_name: str = "InternVL3-8B"
    dit_name: str = "FLUX.1-dev"
    num_mllm_layers: int = 29  # layers 0..28 in InternVL3-8B ablations
    lora_rank: int = 512

    # Dual Layer Aggregator / LAP.
    embed_dim: int = 128
    num_attn_heads: int = 4

    # Multi-stage timestep-aware denoising (Eq. 5), inference defaults.
    tau1: float = 0.95
    tau2: float = 0.85
    cfg_scale: float = 2.5

    # Two-stage training schedule (paper Sec. 4.1).
    stage1_steps: int = 25_000
    stage2_steps: int = 10_000

    # Layer groups used in ablations (InternVL3-8B).
    layer_group_early: tuple[int, int] = (0, 9)
    layer_group_mid: tuple[int, int] = (10, 19)
    layer_group_late: tuple[int, int] = (20, 28)
