"""NVC ERP quality-parameter adaptation (Arai et al., arXiv:2512.20093)."""

from __future__ import annotations

from dataclasses import dataclass
import math

PAPER_ARXIV = "2512.20093"
PAPER_TITLE = (
    "Neural Compression of 360-Degree Equirectangular Videos using Quality Parameter Adaptation"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

# DCVC-RT pretrained settings (Sec. V-A)
LAMBDA_MIN = 1.0
LAMBDA_MAX = 768.0
Q_NUM = 64
LDP_QP_OFFSET_PATTERN = (0, 8, 0, 4, 0, 4, 0, 4)
WS_PSNR_YUV_WEIGHTS = (6, 1, 1)  # (Y, U, V)

# Table I headline
BD_RATE_WS_PSNR_AVG_PCT = -5.20
ENC_FPS_BASELINE = 2.014
ENC_FPS_PROPOSED = 2.008
PROC_TIME_INCREASE_PCT = 0.3


@dataclass
class NvcErpQpaConfig:
    lambda_min: float = LAMBDA_MIN
    lambda_max: float = LAMBDA_MAX
    q_num: int = Q_NUM
    q0: float = 32.0  # base quality parameter (stub encode point)
    latent_channels: int = 16
    erp_height: int = 64
    erp_width: int = 128
    use_interpolation: bool = True  # vs floor-only vector selection


def log_lambda_span(cfg: NvcErpQpaConfig) -> float:
    return math.log(cfg.lambda_max) - math.log(cfg.lambda_min)
