"""SphereUFormer — icosphere U-Transformer for 360° perception (arXiv:2412.06968)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2412.06968"
PAPER_TITLE = "SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
NUM_ENCODER_STAGES = 4
BLOCKS_PER_STAGE = 2
BASE_EMBED_DIM = 32
DEPTH_MAX_M_S2D3D = 10.0
DEPTH_MAX_M_S3D = 5.0
TRAIN_ITERS_S2D3D = 25000
TRAIN_ITERS_S3D = 160000


@dataclass
class SphereUFormerConfig:
    rank: int = 7
    node_type: str = "hex"  # hexasphere = vertices; ico = faces
    c_head: float = 2.0
    c_win: int = 2
    embed_dim: int = BASE_EMBED_DIM
    in_ch: int = 3
    out_ch_depth: int = 1
    num_classes_seg: int = 13
    task: str = "depth"  # depth | segmentation
    rel_grid: int = 7
    use_abs_pos: bool = True
    use_rel_pos: bool = True
