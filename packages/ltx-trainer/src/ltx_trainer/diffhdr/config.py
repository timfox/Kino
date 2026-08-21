"""DiffHDR configuration (Yu et al. arXiv:2604.06161)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2604.06161"
PAPER_TITLE = "DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models"

# Log-Gamma defaults (Sec. 3.2)
LOG_GAMMA_M = 16.0
LOG_GAMMA_GAMMA = 2.2

# Mask detection (Sec. 3.4)
TAU_HIGH = 0.95
TAU_LOW = 0.05
MASK_EMA_ALPHA = 0.7

# Camera noise (Sec. 3.1)
NOISE_SIGMA_S_RANGE = (0.0, 8.5e-4)
NOISE_SIGMA_C_RANGE = (0.0, 1.5e-5)
NOISE_AR1_RHO = 0.5

# Training (Sec. 4.1)
LORA_RANK = 32
TRAIN_STEPS = 10_000
LR = 1e-4
NUM_FRAMES = 81
HDRI_COUNT = 800
SEQUENCES_PER_HDRI = 6
TOTAL_SEQUENCES = 5400

# CFA inference strengths
DEFAULT_ALPHA_OVER = 1.0
DEFAULT_ALPHA_UNDER = 1.0


@dataclass(frozen=True)
class DiffHDRConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    num_frames: int = 8
    image_size: int = 64
    lora_rank: int = LORA_RANK
    log_gamma_m: float = LOG_GAMMA_M
    log_gamma_gamma: float = LOG_GAMMA_GAMMA
    use_mask: bool = True
    use_cfa: bool = True
    use_data_aug: bool = True
    tau_high: float = TAU_HIGH
    tau_low: float = TAU_LOW
    mask_ema_alpha: float = MASK_EMA_ALPHA
