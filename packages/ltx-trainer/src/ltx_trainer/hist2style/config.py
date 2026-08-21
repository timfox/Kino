"""Hist2Style histogram-guided bilateral-grid stylization (arXiv:2606.01819)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Hist2StyleConfig:
    paper_arxiv: str = "arXiv:2606.01819"
    title: str = "Hist2Style: Histogram-Guided Stylization with Bilateral Grids"
    project_url: str = "https://dgalor.github.io/hist2style/"
    params_m: float = 1.5
    teacher: str = "Flux.1 Kontext (distilled subset)"
    training_pairs_m: float = 1.1  # filtered of 1.7M
    vgg_filter_threshold: float = 0.5

    # Bilateral grid (§3.2)
    grid_guidance: int = 8
    grid_height: int = 16
    grid_width: int = 16
    affine_dim: int = 12  # 3×4 per cell

    # Histogram bins per Y/U/V channel
    hist_bins: int = 64
    train_resolution: int = 256

    # Training (§3.5)
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 1127
