"""EDM — ERP-oriented dense kernelized feature matching (arXiv:2502.20685)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2502.20685"
PAPER_TITLE = (
    "EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://jdk9405.github.io/EDM"

ERP_WIDTH = 640
ERP_HEIGHT = 320
GP_TAU = 5.0
GP_SIGMA_N = 0.1
LOSS_LAMBDA_C = 0.01
DEPTH_ALPHA = 0.05
CERTAINTY_THRESH = 0.8
MAX_SAMPLES_EVAL = 5000


@dataclass
class EdmConfig:
    erp_w: int = ERP_WIDTH
    erp_h: int = ERP_HEIGHT
    feature_ch: int = 64
    gp_tau: float = GP_TAU
    gp_sigma_n: float = GP_SIGMA_N
    loss_lambda_c: float = LOSS_LAMBDA_C
    use_spherical_pe: bool = True
    use_geodesic_refine: bool = True
    use_azimuth_aug: bool = True
