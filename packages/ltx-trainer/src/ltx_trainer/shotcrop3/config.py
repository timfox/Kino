"""ShotCrop3 / TSC configuration (arXiv:2606.05635)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GRPOWeights:
    lambda_iou: float = 0.6
    lambda_aes: float = 0.2
    lambda_ratio: float = 0.2


@dataclass(frozen=True)
class PGSThresholds:
    tau_high: float = 0.85
    tau_low: float = 0.6


@dataclass(frozen=True)
class ShotCrop3Params:
    image_size: tuple[int, int] = (1024, 768)
    grpo: GRPOWeights = field(default_factory=GRPOWeights)
    pgs: PGSThresholds = field(default_factory=PGSThresholds)
    target_ratios: tuple[tuple[float, float], ...] = ((4.0, 3.0), (3.0, 4.0))


@dataclass(frozen=True)
class ShotCrop3Config:
    paper_arxiv: str = "arXiv:2606.05635"
    shot_types: tuple[str, ...] = ("medium", "close_up", "establishing")
    dataset_domains: tuple[str, ...] = (
        "travel",
        "street",
        "cinematic",
        "professional",
    )
    train_samples: int = 6400
    test_samples: int = 1200
    base_model: str = "Qwen3-VL-4B"
    params: ShotCrop3Params = field(default_factory=ShotCrop3Params)
    packages: tuple[str, ...] = ("shotcrop3",)
