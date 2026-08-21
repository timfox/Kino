"""Cross360 360° monocular depth (Huang et al., arXiv:2601.17271)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2601.17271"
PAPER_TITLE = "Cross360: 360° Monocular Depth Estimation via Cross Projections Across Scales"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/huangkun101230/Cross360"

NUM_SCALES = 5  # S = 5 decoder levels
NUM_TP_PATCHES_FULL = 26
NUM_TP_PATCHES_INCOMPLETE = 20
TP_FOV_DEG = 72.0
PARAMS_M = 67.84
BACKBONE = "ResNet34"


@dataclass
class Cross360Config:
    height: int = 256
    width: int = 512
    embed_dim: int = 64
    num_heads: int = 4
    num_tp_patches: int = NUM_TP_PATCHES_FULL
    incomplete_fov: bool = False  # M3D / S2D3D pole masking strategy
    max_depth_m: float = 10.0
