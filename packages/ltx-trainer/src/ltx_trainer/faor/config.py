"""FAOR — fast arbitrary-scale ODI super-resolution (arXiv:2502.05902)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2502.05902"
PAPER_TITLE = (
    "Fast Omni-Directional Image Super-Resolution: Adapting the Implicit Image "
    "Function with Pixel and Semantic-Wise Spherical Geometric Priors"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/GingaUL/FAOR"
ERP_H = 1024
ERP_W = 2048
SAFE_BLOCKS = 36
LATENT_DIM = 64


@dataclass
class FaorConfig:
    in_ch: int = 3
    latent_dim: int = LATENT_DIM
    safe_blocks: int = SAFE_BLOCKS
    use_md: bool = True
    use_ms: bool = True
    use_geodesic: bool = True
    lr_size: int = 128
