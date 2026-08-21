"""DeblurNVS configuration (Shi et al. arXiv:2606.01315)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.01315"
PAPER_TITLE = (
    "DeblurNVS: Geometric Latent Diffusion for Novel View Synthesis "
    "from Sparse Motion-Blurred Images"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://github.com/PKU-YuanGroup/DeblurNVS"

# Training data (Sec. III-B)
DL3DV_SCENE_COUNT = 10_000
DL3DV_IMAGES_PER_SCENE = 500
BLUR_WINDOW_SIZES = (5, 7, 9, 11)
INTERPOLATION_RATE = 8
DL3DV_BLUR_PAIR_COUNT = 5_000_000

# Inference (Sec. IV-A)
RESOLUTION = (280, 504)  # H × W
CONTEXT_DENOISE_STEPS = 8
TARGET_DENOISE_STEPS = 12
DEFAULT_CONTEXT_VIEWS = 3

# Loss weights (Sec. III-C, Eq. 14)
LAMBDA_L1 = 1.0
LAMBDA_LPIPS = 1.0
LAMBDA_GAN = 0.75

# Backbone
BACKBONE = "DA3 (Depth Anything 3) + GLD latent diffusion"
LORA_RANK = 16


@dataclass
class DeblurNVSConfig:
    height: int = RESOLUTION[0]
    width: int = RESOLUTION[1]
    latent_channels: int = 64
    context_views: int = DEFAULT_CONTEXT_VIEWS
    context_steps: int = CONTEXT_DENOISE_STEPS
    target_steps: int = TARGET_DENOISE_STEPS
    lora_rank: int = LORA_RANK
    lambda_l1: float = LAMBDA_L1
    lambda_lpips: float = LAMBDA_LPIPS
    lambda_gan: float = LAMBDA_GAN
    blur_window_sizes: tuple[int, ...] = BLUR_WINDOW_SIZES
    interpolation_rate: int = INTERPOLATION_RATE
