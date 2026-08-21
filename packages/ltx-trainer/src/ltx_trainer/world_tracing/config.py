"""World Tracing configuration (Zhang et al. arXiv:2606.13652)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

PAPER_ARXIV = "2606.13652"
PAPER_TITLE = "World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://worldlabs.ai"

DEFAULT_LAYERS = 6
DEFAULT_RESOLUTION = 504
DEFAULT_PATCH = 14
DEFAULT_ODE_STEPS = 20
DEFAULT_WIDTH = 1536
DEFAULT_HEADS = 24
DEFAULT_BLOCKS = 48
TRAINABLE_PARAMS_B = 1.4
TOTAL_PARAMS_B = 1.7
OBJECT_CORPUS_ASSETS = 300_000
OBJECT_RENDER_VIEWS = 17_000_000
DYNAMIC_CLIPS = 16_800


class WTVariant(str, Enum):
    OBJECT = "WT-O"
    SCENE = "WT-S"
    DYNAMIC = "WT-D"


@dataclass
class WTConfig:
    variant: WTVariant = WTVariant.OBJECT
    num_layers: int = DEFAULT_LAYERS
    height: int = 56
    width: int = 56
    patch_size: int = DEFAULT_PATCH
    width_dim: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    moge_feat_dim: int = 64
    ode_steps: int = 2
    lambda_mono: float = 0.1
    object_zscore: tuple[float, float, float] = (0.0, 1.0, 0.0)
    object_zscore_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temporal_frames: int = 1
    temporal_layerscale: float = 1e-5
