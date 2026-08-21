"""FEFormer configuration (Yang et al., arXiv:2605.11434)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2605.11434"
PAPER_TITLE = "FEFormer: Frequency-enhanced Vision Transformer for Generic Knowledge Extraction and Adaptive Feature Fusion in Volumetric Medical Image Segmentation"

BASE_CHANNELS = 64
MLP_RATIO = 4
PATCH_SIZE = 96
STEM_DOWNSAMPLE = 4
PARAMS_M = 18.54
FLOPS_G = 39.13


@dataclass
class FEFormerConfig:
    in_channels: int = 1
    num_classes: int = 15
    base_channels: int = BASE_CHANNELS
    mlp_ratio: int = MLP_RATIO
    dw_kernel: int = 7
    image_size: int = 32  # stub training size; paper uses 96


DATASETS = (
    "amos2022",
    "hepatic_vessel_tumor",
    "brain_tumor",
    "flare",
)
