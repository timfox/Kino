"""UAV-OVO configuration (arXiv:2605.25615)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UAVOVOConfig:
    """Reference hyperparameters from the UAV-OVO / LATER paper."""

    paper_arxiv: str = "arXiv:2605.25615"
    num_classes: int = 155
    train_view_max: float = 30.0
    isolation_view_min: float = 30.0
    isolation_view_max: float = 40.0
    ood_view_min: float = 40.0
    train_videos: int = 13872
    id_test_videos: int = 3100
    isolation_videos: int = 3275
    ood_test_videos: int = 3100
    ood_topup_videos: int = 981
    total_videos: int = 23347
    backbone: str = "MViTv2-B"
    lora_rank: int = 16
    lora_gamma: float = 1.0
    recenter_alpha: float = 1.0
    feature_dim: int = 256
    svd_keep_ratio: float = 1e-4
    queue_size: int = 32
