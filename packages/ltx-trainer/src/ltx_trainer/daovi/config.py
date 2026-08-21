"""DAOVI — distortion-aware omnidirectional video inpainting (arXiv:2509.00396)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2509.00396"
PAPER_TITLE = "DAOVI: Distortion-Aware Omnidirectional Video Inpainting"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

ODV360_TRAIN = 210
ODV360_VAL = 20
ODV360_TEST = 20
ODV360_FRAMES = 100
TRAIN_RESOLUTION = (152, 304)  # H x W
GEODESIC_THRESHOLD_DEG = 0.4
TRAIN_ITERATIONS = 80_000
BATCH_SIZE = 8
LEARNING_RATE = 1.5e-4


@dataclass
class DaoviConfig:
    height: int = 152
    width: int = 304
    num_frames: int = 5
    hidden_dim: int = 64
    geodesic_eps_deg: float = GEODESIC_THRESHOLD_DEG
    use_gfcip: bool = True
    use_odafp: bool = True
    encoder_layers: int = 9
    decoder_layers: int = 4
