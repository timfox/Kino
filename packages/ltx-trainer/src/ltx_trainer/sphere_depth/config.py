"""Sphere-Depth configuration (Gazzeh et al. arXiv:2604.23432)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2604.23432"
PAPER_TITLE = (
    "Sphere-Depth: A Benchmark for Depth Estimation Methods "
    "with Varying Spherical Camera Orientations"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/sgazzeh/Sphere_depth"

# 8 Garmin VIRB 360 indoor panoramas + 37 landmarks (paper Sec. 3)
NUM_IMAGES = 8
NUM_LANDMARKS = 37
SMALL_DEFORM_DEG = 1.0
HIGH_DEFORM_DEG = 15.0

BENCHMARK_MODELS = (
    "ACDNet",
    "DepthAnywhere",
    "BiFuse++",
    "SliceNet",
    "DepthAnythingV2",
)

# Pitch/roll grids for sensitivity study (Sec. 3.3)
LARGE_POSE_GRID_DEG = tuple(range(-40, 41, 10))
SMALL_POSE_GRID_DEG = tuple(x * 0.5 for x in range(-4, 5))  # -2 .. +2 step 0.5

DEPTH_CLIP_M = (0.0, 10.0)


@dataclass
class SphereDepthConfig:
    erp_height: int = 256
    erp_width: int = 512
    train_landmark_ratio: float = 0.6
    disparity_alpha: float = 1.0
    disparity_eps: float = 1e-3
    cubemap_face_size: int = 128
