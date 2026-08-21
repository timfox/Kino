"""PriOr-Flow — primitive panoramic optical flow with orthogonal view (arXiv:2506.23897)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2506.23897"
PAPER_TITLE = "PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/longliangLiu/PriOr-Flow"

ORTHOGONAL_ROTATION_DEG = 90.0
DCCL_RADIUS = 1
REFINE_ITERATIONS = 12
LOSS_GAMMA = 0.8
LEARNING_RATE = 1e-4
MPF_TRAIN_STEPS = 60_000
FLOWSCAPE_TRAIN_STEPS = 100_000
POLAR_LATITUDE_DEG = 45.0


@dataclass
class PriorFlowConfig:
    height: int = 128
    width: int = 256
    feature_dim: int = 64
    dccl_radius: int = DCCL_RADIUS
    num_iterations: int = 4
    use_dccl: bool = True
    use_oddc: bool = True
    orthogonal_axis: str = "x"
