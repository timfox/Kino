"""MTPano configuration (Zhang et al. arXiv:2602.05330)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2602.05330"
PAPER_TITLE = (
    "MTPano: Multi-Task Panoramic Scene Understanding via "
    "Label-Free Integration of Dense Prediction Priors"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/Evergreen0929/MTPano"

INPUT_SIZE = (512, 1024)  # H × W
TRAIN_PANORAMAS = 140_884
PATCHES_PER_PANO = 32
EMBED_DIM = 512
BACKBONE = "DINOv3-Large"

# Main + auxiliary task loss weights (Sec. 4.1)
LAMBDA_SEM = LAMBDA_DEPTH = LAMBDA_NORM = 1.0
LAMBDA_AUX = 0.003
WARMUP_STEPS = 1000
TRAIN_ITERS = 100_000


@dataclass
class MTPanoConfig:
    height: int = 256
    width: int = 512
    embed_dim: int = 64
    bridge_dim: int = 128
    fov_deg: float = 90.0
