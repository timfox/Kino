"""CAFNet: cross-attentive half-truth deepfake detection (arXiv:2605.29531)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeepfakeClass(str, Enum):
    REAL = "real"
    FAKE = "fake"
    HALF_TRUTH = "half_truth"


@dataclass
class CafNetConfig:
    paper_arxiv: str = "arXiv:2605.29531"
    github: str = "https://github.com/ssutharya/Audio_Deepfake_Detection"
    sample_rate_hz: int = 16000
    clip_seconds: float = 4.0
    mfcc_coeffs: int = 40
    lfcc_coeffs: int = 40
    chroma_bins: int = 12
    feature_frames: int = 251
    hidden_channels: int = 128
    attn_heads: int = 8
    bilstm_units: int = 64
    cafnet_params: int = 576_414
    mfaan_params: int = 322_562
    batch_size: int = 64
    learning_rate: float = 5e-4
    aux_loss_weight: float = 0.4
    temp_loss_weight: float = 0.3
    class_weights: tuple[float, float, float] = (1.622, 0.811, 0.568)
    random_seed: int = 42
