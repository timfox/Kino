"""PAR — Panoramic AutoRegressive model (arXiv:2505.16862)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2505.16862"
PAPER_TITLE = "Conditional Panoramic Image Generation via Masked Autoregressive Modeling"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_PAGE = "https://wang-chaoyang.github.io/project/par"

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 1024
TRAIN_ITERATIONS = 20_000
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
CFG_SCALE = 5.0
LAMBDA_CONSISTENCY = 0.1
DEFAULT_PAD_RATIO = 0.125
AR_SAMPLING_STEPS = 64
MLP_DENOISE_STEPS = 25


@dataclass
class PARConfig:
    height: int = 64
    width: int = 128
    latent_dim: int = 32
    text_dim: int = 32
    num_patches: int = 64
    mask_ratio: float = 0.5
    lambda_consistency: float = LAMBDA_CONSISTENCY
    pad_ratio: float = DEFAULT_PAD_RATIO
    model_size: str = "0.3B"
