"""HypeVPR — hyperbolic P2E visual place recognition (arXiv:2506.04764)."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2506.04764"
PAPER_TITLE = "HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://suhan-woo.github.io/HypeVPR/"

DESCRIPTOR_DIM_P2E = 768
DESCRIPTOR_DIM_P2P = 2048
PANO_WIDTH_RATIO = 8
DEFAULT_CURVATURE = 1.0
TRIPLET_MARGIN = 0.1


@dataclass
class HypeVPRConfig:
    """Stub defaults (P2E: 224 query, 8× width panorama, L=4 avoids overlap)."""

    query_size: int = 64
    pano_width_ratio: int = PANO_WIDTH_RATIO
    hierarchy_levels: int = 4
    descriptor_dim: int = 32
    curvature: float = DEFAULT_CURVATURE
    gem_p: float = 3.0
    triplet_margin: float = TRIPLET_MARGIN
    top_k_prime: int = 8
    retrieval_levels: tuple[int, ...] = (1, 4)
    level_weights: dict[int, float] = field(default_factory=lambda: {1: 1.0, 4: 0.5})
