"""Configuration for fast LPM MCMC (arXiv:2605.30134)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LPMMCMCConfig:
    paper_arxiv: str = "arXiv:2605.30134"
    latent_dim: int = 2
    taylor_order: int = 4  # κ; use 1 for Algorithm 10
    block_width: float = 0.1  # b
    num_blocks: int = 16  # K after partition
    gaussian_link: bool = True
    beta0: float = 0.1
    beta1: float = 0.7
    sigma: float = 0.6
    mg: float = 1.0  # Taylor coefficient Mg
    bg: float = 2.0  # 2 / ρ_g
    epsilon_post: float = 0.01
    default_sweeps: int = 100
