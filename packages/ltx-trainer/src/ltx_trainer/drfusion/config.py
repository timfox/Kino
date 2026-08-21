"""Hyperparameters for DRFusion (Li et al., ICML 2026 / arXiv:2605.25775)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DRFusionConfig:
    """Paper defaults (Sec. 4.1)."""

    paper_arxiv: str = "arXiv:2605.25775"
    code_url: str = "https://github.com/xhhaoyan/DRFusion"
    backbone: str = "3D-DiT (frozen) + Condition Adapter"

    stage1_epochs: int = 50
    stage1_batch: int = 16
    stage2_epochs: int = 100
    stage2_batch: int = 2
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"

    ddim_steps: int = 50
    history_window: int = 8
    guidance_scale: float = 2.0
    stabilize_sigma: float = 0.02
    latent_refine_every: int = 5

    lambda_p: float = 1.0
    lambda_s: float = 1.0
    lambda_g: float = 1.0
    lambda_i: float = 1.0
    lambda_reg: float = 0.1

    datasets: tuple[str, ...] = ("HDO", "M3SVD", "NOT-156", "VTMOT")
