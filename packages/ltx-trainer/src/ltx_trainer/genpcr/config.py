"""Generative point cloud registration (Jiang et al., arXiv:2512.09407)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2512.09407"
PAPER_TITLE = (
    "Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

FUSION_WEIGHT_DEFAULT = 0.5
COLOR_FEAT_DIM_DEFAULT = 64
FINETUNE_SAMPLES_DEPTH = 3000
FINETUNE_SAMPLES_LIDAR = 10000
GEO_FEAT_DIM = 32


@dataclass
class GenPcrConfig:
    mode: str = "depth"  # depth | lidar
    image_height: int = 64
    image_width: int = 64
    geo_dim: int = GEO_FEAT_DIM
    color_dim: int = COLOR_FEAT_DIM_DEFAULT
    fusion_weight: float = FUSION_WEIGHT_DEFAULT
    coupled_denoise: bool = True
    num_points: int = 512
