"""LumaFlux configuration (Saini et al. arXiv:2604.02787)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2604.02787"
PAPER_TITLE = "LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers"

L_MAX_NITS = 10_000.0
PEAK_MASTER_NITS = 1_000.0

# BT.2020 luma coefficients (Eq. 6)
M2020_LUMA = (0.2627, 0.6780, 0.0593)

# Training (Sec. 6.1)
TRAIN_ITERATIONS = 200_000
BATCH_SIZE = 16
LR = 1e-4
WARMUP_STEPS = 5_000
INFERENCE_ODE_STEPS = 40

# Loss weights (Eq. 18) — typical relative weighting
LAMBDA_LUMA = 1.0
LAMBDA_RGB = 1.0
LAMBDA_SPLINE_SMOOTH = 0.1

# Adapter defaults
LORA_RANK = 8
NUM_SPECTRAL_BANDS = 8
NUM_DIT_BLOCKS = 4


@dataclass(frozen=True)
class LumaFluxConfig:
    hidden_dim: int = 64
    embed_dim: int = 128
    image_size: int = 64
    lora_rank: int = LORA_RANK
    num_blocks: int = NUM_DIT_BLOCKS
    num_spectral_bands: int = NUM_SPECTRAL_BANDS
    use_pga: bool = True
    use_pcm: bool = True
    use_coupler: bool = True
    use_spectral_gating: bool = True
    rqs_knots: int = 8
    lambda_luma: float = LAMBDA_LUMA
    lambda_rgb: float = LAMBDA_RGB
    lambda_spline: float = LAMBDA_SPLINE_SMOOTH
