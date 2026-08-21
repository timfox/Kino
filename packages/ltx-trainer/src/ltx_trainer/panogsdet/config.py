"""PanoGSDet configuration (Ning et al. arXiv:2605.14601)."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2605.14601"
PAPER_TITLE = (
    "Towards Accurate Single Panoramic 3D Detection: A Semantic Gaussian Centric Approach"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

STRUCTURED3D_CATEGORIES = (
    "bed",
    "chair",
    "sofa",
    "table",
    "desk",
    "dresser",
    "cabinet",
    "fridge",
    "sink",
    "lamp",
    "bathtub",
)
NUM_CLASSES = len(STRUCTURED3D_CATEGORIES)

DEFAULT_ERP_HEIGHT = 256
DEFAULT_ERP_WIDTH = 512


@dataclass
class PanoGSDetConfig:
    height: int = DEFAULT_ERP_HEIGHT
    width: int = DEFAULT_ERP_WIDTH
    feature_dim: int = 32
    num_classes: int = NUM_CLASSES
    r_max: float = 0.15
    center_step_scale: float = 0.05
    cov_beta: float = 0.1
    cov_eta: float = 0.2
    opt_blocks: int = 2
    voxel_size: float = 0.08
    freeze_depth_branch: bool = True
    depth_feature_dim: int = 32
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-3
    grad_clip: float = 35.0
    train_epochs: int = 12
    foreground_class_ids: tuple[int, ...] = field(default_factory=lambda: tuple(range(NUM_CLASSES)))
