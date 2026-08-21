"""ChWDTA learned image compression (Fu et al., arXiv:2606.00111)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ChwdtaConfig:
    paper_arxiv: str = "arXiv:2606.00111"
    title: str = (
        "ChWDTA: Channel-wise Wavelet-Domain Transformer Attention "
        "and Entropy Modeling for Learned Image Compression"
    )
    repo_url: str = "https://github.com/fengyurenpingsheng/ChWDTA-Learned-image-compression"

    # Latent / hyperprior channels (§IV)
    latent_channels: int = 320
    hyper_channels: int = 192
    stage_channels: Tuple[int, ...] = (96, 144, 256, 192, 192)
    transformer_layers: Tuple[int, int, int] = (1, 2, 12)

    # ChWP entropy: 4 subbands × 2 slices = 8 (default)
    chwp_subbands: Tuple[str, ...] = ("LL", "LH", "HL", "HH")
    entropy_slices_8: int = 8
    entropy_slices_4: int = 4

    # BD-rate vs VTM 9.1 anchor (Table I, %)
    bd_rate_kodak_8: float = -17.82
    bd_rate_clic_8: float = -19.15
    bd_rate_tecnick_8: float = -22.56
    bd_rate_kodak_4: float = -17.24
    bd_rate_clic_4: float = -18.42
    bd_rate_tecnick_4: float = -21.71

    # Complexity Kodak 768×512 (Table I)
    latency_8_ms: int = 225
    latency_4_ms: int = 187
    params_8_m: float = 189.1
    params_4_m: float = 113.0
    kmacs_8: float = 1204.68
    kmacs_4: float = 913.65

    # Training (§IV)
    mse_lambdas: Tuple[float, ...] = (0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.0500)
    train_epochs: int = 70
    crop_sizes: Tuple[int, int] = (256, 512)

    # DCAE comparison margins (8-slice vs DCAE)
    dcae_bd_kodak: float = -16.98
