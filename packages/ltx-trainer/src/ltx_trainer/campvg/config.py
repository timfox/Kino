"""CamPVG — camera-controlled panoramic video (arXiv:2509.19979)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2509.19979"
PAPER_TITLE = "CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
DOI = "10.1145/3757377.3763990"

ERP_HEIGHT = 256
ERP_WIDTH = 512
NUM_FRAMES = 16
EPIPOLAR_SAMPLES_K = 250
TRAIN_EPOCHS = 300


@dataclass
class CamPVGConfig:
    height: int = ERP_HEIGHT
    width: int = ERP_WIDTH
    num_frames: int = NUM_FRAMES
    hidden_dim: int = 64
    epipolar_k: int = EPIPOLAR_SAMPLES_K
    use_pano_plucker: bool = True
    use_spherical_epipolar: bool = True
    random_cond_frame: bool = True
