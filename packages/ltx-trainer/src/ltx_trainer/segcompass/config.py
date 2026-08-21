"""Configuration for SegCompass (arXiv:2605.22658)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MLLMBackbone(str, Enum):
    LLAVA_1_5_7B = "llava-1.5-7b"
    LLAVA_1_5_13B = "llava-1.5-13b"
    QWEN2_5_VL_7B = "qwen2.5-vl-7b"


@dataclass
class SAEExtractLayer:
    backbone: str
    layer_index: int
    num_layers: int


SAE_LAYERS: tuple[SAEExtractLayer, ...] = (
    SAEExtractLayer("llava-1.5-7b", 16, 32),
    SAEExtractLayer("llava-1.5-13b", 16, 40),
    SAEExtractLayer("qwen2.5-vl-7b", 13, 28),
)


@dataclass
class SegCompassConfig:
    paper_arxiv: str = "arXiv:2605.22658"
    code_url: str = "https://github.com/ZhenyuLU-Heliodore/SegCompass"
    policy_hidden_d: int = 4096
    sae_dim: int = 65536
    concept_dim: int = 256
    max_slots: int = 6
    grpo_group_size: int = 8
    grpo_clip_ratio: float = 0.2
    grpo_kl_beta: float = 0.2
    lambda_seg: float = 1.0
    lambda_conf: float = 0.2
    lambda_dice: float = 0.6
    sae_alpha: float = 0.1
    reward_format: float = 0.3
    reward_segmentation: float = 0.7
    default_backbone: MLLMBackbone = MLLMBackbone.QWEN2_5_VL_7B
    vision_backbone: str = "SAM-ViT-H"
    mllm_lr: float = 2e-6
    lr_multiplier_codebook: float = 25.0
    lr_multiplier_slot_mapper: float = 80.0
    lr_multiplier_mask_decoder: float = 10.0
    training_steps: int = 24252
    obelics_sae_samples: int = 200_000
