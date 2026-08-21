"""VUGA configuration (Yan et al. arXiv:2604.23953)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2604.23953"
PAPER_TITLE = (
    "Viewport-Unaware Blind Omnidirectional Image Quality Assessment: "
    "A Unified and Generalized Approach"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/KangchengWu/VUGA"

OIQA_DATABASES = (
    "CVIQ",
    "OIQA",
    "MVAQD",
    "IQA-ODI",
    "OSIQA",
    "AIGCOIQA",
    "JUFE-10K",
    "OIQ-10K",
)

IQA_DATABASES = ("KADID-10k", "KonIQ-10k", "LIVE", "SPAQ", "TID2013")


@dataclass
class VUGAConfig:
    input_size: int = 224
    high_res: int = 1024
    in_channels: int = 3
    stage_dims: tuple[int, ...] = (96, 192, 384, 768)
    cmp_dim: int = 512
    train_lr: float = 1e-4
    weight_decay: float = 1e-4
    train_epochs: int = 10
    batch_size: int = 8
    freeze_backbone: bool = True
