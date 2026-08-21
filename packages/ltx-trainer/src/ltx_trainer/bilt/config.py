"""BiLT-Autoencoder configuration (Hohmann, arXiv:2605.11829)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2605.11829"
PAPER_TITLE = (
    "Bin Latent Transformer (BiLT): A shift-invariant autoencoder for "
    "calibration-free spectral unmixing of turbid media"
)

SPECTRAL_POINTS = 150
WAVELENGTH_MIN_NM = 500
WAVELENGTH_MAX_NM = 800
WAVELENGTH_STEP_NM = 2

NUM_SAMPLES = 496
NUM_PROBES = 16
FEATURE_DIM = 128
LATENT_DIM = 3  # IL scatterer + red ink + black ink
DECODER_OUT = 300  # 150 × 2 interleaved μ_a, μ'_s
PARAMS_M = 1.3

# Loss coefficients (Sec. 2.2.2)
MU_A_WEIGHT = 20.0
LOG_LOSS_ALPHA = 0.1
SWAP_LOSS_BETA = 1.0
LATENT_EXCL_LAMBDA = 0.1
LOG_EPS = 1e-6

# Augmentation (Sec. 2.2.3)
MAX_SHIFT_BANDS = 7
MAX_NOISE = 0.03
AUGMENT_RATIO = 0.65


@dataclass
class BiLTConfig:
    spectral_points: int = SPECTRAL_POINTS
    num_probes: int = NUM_PROBES
    feature_dim: int = FEATURE_DIM
    latent_dim: int = LATENT_DIM
    conv_kernel: int = 23
    conv_filters: int = 128
    ffn_dim: int = 256
    bottleneck_dim: int = 64
    num_heads: int = 8
    use_importance_gating: bool = True
    l1_latent: float = 1e-4
