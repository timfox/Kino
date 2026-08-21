"""S3PO — 360° video super-resolution (arXiv:2506.14803)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2506.14803"
PAPER_TITLE = (
    "Omnidirectional Video Super-Resolution using Deep Learning (S3PO)"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

SCALE_FACTOR = 4
WSS_L1_BETA = 1.0
LEARNING_RATE = 1e-4
VIMEO90K_EPOCHS = 70
V360VDS_EPOCHS = 75
NUM_INPUT_FRAMES = 3
NUM_DUAL_DUCT_BLOCKS = 10
PARAMS_M = 8.89
ERP_TRAIN_HEIGHT = 360
ERP_TRAIN_WIDTH = 480


@dataclass
class S3POConfig:
    lr_height: int = 90
    lr_width: int = 120
    feature_dim: int = 64
    scale: int = SCALE_FACTOR
    num_duct_blocks: int = 2
    use_attention: bool = True
    use_cyclic: bool = False
    wss_beta: float = WSS_L1_BETA
