"""PanoWorld-X — explorable panoramic video via sphere-aware DiT (arXiv:2509.24997)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2509.24997"
PAPER_TITLE = "PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://yuyangyin.github.io/PanoWorld-X/"

# PanoExplorer dataset (Sec. 3.2)
PANOEXPLORER_VIDEOS = 116_759
PANOEXPLORER_SCENES = 504
MIN_TRAJECTORY_METERS = 18.0
FRAME_STRIDE_METERS = 0.10
NUM_FRAMES = 49
ERP_HEIGHT = 480
ERP_WIDTH = 960  # 1:2 aspect after resize from 480×720

# Sphere-aware attention
SPHERE_DISTANCE_THRESHOLD = 0.35  # τ in Eq. 6 (radians on unit sphere, stub default)


@dataclass
class PanoWorldXConfig:
    height: int = ERP_HEIGHT
    width: int = ERP_WIDTH
    num_frames: int = NUM_FRAMES
    patch_size: int = 16
    hidden_dim: int = 64
    sphere_threshold: float = SPHERE_DISTANCE_THRESHOLD
    use_exp_branch: bool = True
    use_sphere_branch: bool = True
    freeze_global_attn: bool = True
    plucker_dim: int = 6
