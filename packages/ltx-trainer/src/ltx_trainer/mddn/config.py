"""MDDN omnidirectional image SR (Yang et al., arXiv:2512.17343)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2512.17343"
PAPER_TITLE = (
    "Multi-level distortion-aware deformable network for omnidirectional image super-resolution"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

# Training defaults (Sec. 4.1.2)
HIDDEN_DIM = 156
NUM_MDDB = 6
NUM_MDDL_PER_MDDB = 6
PATCH_SIZE = 256
SCALE_FACTORS = (4, 8, 16)
LOW_RANK_RANK = 8


@dataclass
class MddnConfig:
    in_channels: int = 3
    hidden_dim: int = HIDDEN_DIM
    scale: int = 4
    erp_height: int = 64
    erp_width: int = 128
    low_rank_rank: int = LOW_RANK_RANK
    use_mff: bool = True  # vs pixel-wise addition (Table 5)
    branches: tuple[int, ...] = (1, 2, 3)  # DDCA + D4C dilations
