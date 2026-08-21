"""GSpaRC — Gaussian splatting for real-time RF channel reconstruction (arXiv:2511.22793)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2511.22793"
PAPER_TITLE = "GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

# Table 2 defaults
LAMBDA_L1_SSIM = 0.2
CONFIDENCE_ALPHA = 0.3
CONFIDENCE_THRESHOLD = 0.59
PILOT_FREE_FRACTION = 0.71
EMISSION_HIDDEN = 16
CONFIDENCE_HIDDEN = 32


@dataclass
class GSpaRCConfig:
    spectrum_height: int = 90  # ~1° hemispherical grid stub
    spectrum_width: int = 180
    num_gaussians: int = 128
    emission_hidden: int = EMISSION_HIDDEN
    confidence_hidden: int = CONFIDENCE_HIDDEN
    lambda_l1_ssim: float = LAMBDA_L1_SSIM
    confidence_alpha: float = CONFIDENCE_ALPHA
    use_distance_attenuation: bool = True  # 1/d in Eq. 3
    objective: str = "spectrum"  # spectrum | channel
