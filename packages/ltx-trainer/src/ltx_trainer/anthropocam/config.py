"""AnthropoCam Anthropocene NST mobile deployment (arXiv:2601.21141)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TextureClass = Literal["modular", "filamentous", "mixed"]

PAPER_ARXIV = "arXiv:2601.21141"
PAPER_TITLE = (
    "Optimization and Mobile Deployment for Anthropocene Neural Style Transfer"
)
PAPER_URL = "https://arxiv.org/abs/2601.21141"

# VGG-16 layer names used in the paper (Gatys / Dumoulin pipeline).
CONTENT_LAYER = "conv3_3"
MODULAR_STYLE_LAYERS = ("conv4_2", "conv4_3")
FILAMENTOUS_STYLE_LAYERS = ("conv2_2", "conv3_1")
DEFAULT_STYLE_LAYERS = ("conv1_2", "conv2_2", "conv3_3", "conv4_3")

# Empirical optimums from §5.
OPTIMAL_STYLE_WEIGHT = 5.0
CONTENT_WEIGHT = 1.0
OPTIMAL_EPOCHS = 10
OPTIMAL_BATCH_SIZE = 8
TV_WEIGHT_DEFAULT = 1e-5

# Mobile resolution trade-off (§5.3).
MOBILE_RESOLUTION = (1280, 2276)
HIGH_RES = (1920, 3416)
LOW_RES = (540, 960)
LATENCY_TARGET_S = (3.0, 5.0)


@dataclass
class AnthropoCamConfig:
    paper_arxiv: str = PAPER_ARXIV
    title: str = PAPER_TITLE
    paper_url: str = PAPER_URL
    backbone: str = "VGG-16"
    inference_mode: str = "feed_forward"  # Johnson et al.; not iterative Gatys

    content_layer: str = CONTENT_LAYER
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    texture_class: TextureClass = "mixed"

    content_weight: float = CONTENT_WEIGHT
    style_weight: float = OPTIMAL_STYLE_WEIGHT
    tv_weight: float = TV_WEIGHT_DEFAULT

    train_epochs: int = OPTIMAL_EPOCHS
    train_batch_size: int = OPTIMAL_BATCH_SIZE
    train_resolution: int = 256

    mobile_width: int = MOBILE_RESOLUTION[0]
    mobile_height: int = MOBILE_RESOLUTION[1]
    latency_target_s: tuple[float, float] = LATENCY_TARGET_S

    num_styles: int = 4  # conditional instance norm slots (Dumoulin)
    style_dataset_homogeneous: bool = True

    stack_frontend: str = "React Native"
    stack_backend: str = "Flask GPU server"

    layer_weights: dict[str, float] = field(
        default_factory=lambda: {layer: 1.0 for layer in DEFAULT_STYLE_LAYERS}
    )

    def style_layers_for_texture(self) -> tuple[str, ...]:
        if self.texture_class == "modular":
            return MODULAR_STYLE_LAYERS
        if self.texture_class == "filamentous":
            return FILAMENTOUS_STYLE_LAYERS
        return self.style_layers
