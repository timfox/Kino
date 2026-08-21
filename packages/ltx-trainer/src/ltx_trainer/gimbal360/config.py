"""Gimbal360 configuration (Lu et al. arXiv:2603.23179)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2603.23179"
PAPER_TITLE = (
    "Gimbal360: Differentiable Auto-Leveling for Canonicalized "
    "360° Panoramic Image Completion"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://orange-3dv-team.github.io/Gimbal360Perspective"

# Horizon360 (Sec. 4)
HORIZON360_SIZE = 20_000
ERP_TRAIN_SIZE = (960, 1920)
PERSPECTIVE_PER_PANO = 3

# Training (Sec. 5.1, Appendix A)
BASE_MODEL = "Flux.1-fill-dev"
TRAIN_STEPS = 10_000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
LORA_RANK = 64
LAMBDA_SHIFT = 0.5
LAMBDA_FLOW = 0.1

# Inference (Appendix A.2)
DENOISE_STEPS = 30
CFG_SCALE = 30.0


@dataclass
class Gimbal360Config:
    erp_height: int = 96
    erp_width: int = 192
    perspective_height: int = 64
    perspective_width: int = 64
    latent_channels: int = 4
    lambda_shift: float = LAMBDA_SHIFT
    lambda_flow: float = LAMBDA_FLOW
    mixture_lambda: float = 0.7
