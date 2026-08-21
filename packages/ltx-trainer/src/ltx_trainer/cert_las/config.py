"""Configuration for Cert-LAS (arXiv:2605.29809)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CertLASConfig:
    paper_arxiv: str = "arXiv:2605.29809"
    backbone: str = "Stable Diffusion v1.4"
    class_prompts: tuple[str, str] = ("a photo of a cat", "a photo of a dog")
    watermark_class_index: int = 0  # cat prompt is ˜y; target pushes toward dog
    target_lambda: float = 0.55  # q* = [1-lambda, lambda]
    sigma_uniform: float = 0.01
    sigma_scale_k: float = 1.0  # inference-time smoothing multiplier
    omega0: float = 5e-5
    lfs_estimation_steps: int = 2000
    verify_M: int = 100
    verify_N: int = 100
    significance_alpha: float = 0.05
    rp_zeta_upper: float = 0.125
    exponential_Tg: int = 500
    m_noise_min: int = 1
    m_noise_max: int = 32
    fold_role: str = "diffusion_mov_provenance"
