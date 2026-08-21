"""VGG layer presets for Anthropocene texture classes (§5.1.1)."""

from __future__ import annotations

from typing import Literal

from ltx_trainer.anthropocam.config import (
    CONTENT_LAYER,
    FILAMENTOUS_STYLE_LAYERS,
    MODULAR_STYLE_LAYERS,
    DEFAULT_STYLE_LAYERS,
)

TextureClass = Literal["modular", "filamentous", "mixed"]


def content_layer() -> str:
    return CONTENT_LAYER


def style_layers_for(texture_class: TextureClass) -> tuple[str, ...]:
    if texture_class == "modular":
        return MODULAR_STYLE_LAYERS
    if texture_class == "filamentous":
        return FILAMENTOUS_STYLE_LAYERS
    return DEFAULT_STYLE_LAYERS


def layer_depth_index(layer: str) -> int:
    """Map convX_Y to monotonic depth for proxy feature nets."""
    block, sub = layer.replace("conv", "").split("_")
    return int(block) * 10 + int(sub)
