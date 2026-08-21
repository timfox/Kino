"""Configuration for DIVA (arXiv:2605.25328)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DIVAConfig:
    paper_arxiv: str = "arXiv:2605.25328"
    github: str = "https://github.com/Jayyy-H/DIVA"

    # Middle layers where factorization / MI objectives apply (Show-o default in paper).
    mid_layer_start: int = 8
    mid_layer_end: int = 18

    # Masked inpainting for generation flow (Sec. 3.1).
    mask_ratio_min: float = 0.2
    mask_ratio_max: float = 0.6

    # Stage 2 loss weights (Table 8; λ_uni best at 0.6 in Table 4).
    lambda_und: float = 1.0
    lambda_gen: float = 1.0
    lambda_orth: float = 0.2  # L⊥ Eq. (9)
    lambda_uni: float = 0.6
    lambda_sha: float = 0.6

    # Low-rank logit readout rank (Appendix B.1).
    readout_rank: int = 24

    # Toy smoke dimensions.
    hidden_dim: int = 64
    factor_dim: int = 32
    temperature: float = 0.07

    umm_backbones: tuple[str, ...] = ("Nexus-Gen", "Show-o", "Liquid")
