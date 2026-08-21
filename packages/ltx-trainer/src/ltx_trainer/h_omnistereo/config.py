"""H-OmniStereo configuration (Jiang et al. arXiv:2605.14963)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2605.14963"
PAPER_TITLE = "H-OmniStereo: Zero-Shot Omnidirectional Stereo Matching with Heading-Aligned Normal Priors"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://github.com/JIANG-CX/H-OmniStereo"

DEFAULT_ERP_WIDTH = 1024
DEFAULT_ERP_HEIGHT = 512
DEFAULT_MAX_DISPARITY = 256
DEFAULT_REFINE_ITERS = 22
DATASET_STEREO_PAIRS = 2_800_000
DATASET_NORMAL_IMAGES = 8_400_000


@dataclass
class HOmniStereoConfig:
    width: int = DEFAULT_ERP_WIDTH
    height: int = DEFAULT_ERP_HEIGHT
    max_disparity: int = DEFAULT_MAX_DISPARITY
    refine_iters: int = DEFAULT_REFINE_ITERS
    feature_dim: int = 64
    cost_groups: int = 8
    gamma_disp: float = 0.9
    baseline_m: float = 0.2
    train_crop_w: int = 256
    train_crop_h: int = 256
    freeze_normal_priors_in_stereo: bool = True
