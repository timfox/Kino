"""360Anything geometry-free perspective→360° (Wu et al., arXiv:2601.16192)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2601.16192"
PAPER_TITLE = "360Anything: Geometry-Free Lifting of Images and Videos to 360°"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://360anything.github.io"

IMAGE_BACKBONE = "FLUX.1-dev"
VIDEO_BACKBONE = "Wan2.1-14B"
IMAGE_ERP_SIZE = (1024, 2048)  # H×W
VIDEO_ERP_SIZE = (512, 1024)
VIDEO_FRAMES = 81
CLE_PAD_FRAC = 1 / 8  # w′ = W/8


@dataclass
class Anything360Config:
    mode: str = "image"  # image | video
    erp_height: int = 128
    erp_width: int = 256
    pers_height: int = 64
    pers_width: int = 64
    latent_channels: int = 16
    patch_size: int = 2
    num_frames: int = 9  # stub video length
    use_circular_latent: bool = True
    sequence_concat: bool = True  # vs channel-concat baseline
