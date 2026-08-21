"""ErpGS — equirectangular 3D Gaussian splatting (arXiv:2505.19883)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2505.19883"
PAPER_TITLE = "ErpGS: Equirectangular Image Rendering Enhanced with 3D Gaussian Regularization"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

DEFAULT_ERP_HEIGHT = 800
DEFAULT_ERP_WIDTH = 1600
OPT_ITERATIONS = 30_000
REG_START_ITER = 10_000

LAMBDA_DN = 0.01
LAMBDA_F = 100.0
LAMBDA_S = 0.01
LAMBDA_SSIM = 0.2


@dataclass
class ErpGSConfig:
    height: int = 128
    width: int = 256
    num_gaussians: int = 64
    feature_dim: int = 32
    lambda_dn: float = LAMBDA_DN
    lambda_f: float = LAMBDA_F
    lambda_s: float = LAMBDA_S
    lambda_ssim: float = LAMBDA_SSIM
    reg_start_iter: int = REG_START_ITER
