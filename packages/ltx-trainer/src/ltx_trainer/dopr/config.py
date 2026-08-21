"""DoPr configuration (arXiv:2606.06418)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GpKind = Literal["sgd", "adam", "adamw", "muon", "signum", "adamuon"]


@dataclass(frozen=True)
class DoPrConfig:
    paper_arxiv: str = "arXiv:2606.06418"
    gp: GpKind = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    damping: float = 1e-4
    damping_mode: Literal["trace", "fixed"] = "trace"
    use_grad_ema: bool = False
    use_cov_ema: bool = False
    ema_beta: float = 0.9
    packages: tuple[str, ...] = (
        "preconditioning",
        "gp",
        "ttf",
        "feature_learning",
        "invariance",
        "metrics",
    )


@dataclass(frozen=True)
class SyntheticFeatureConfig:
    d_y: int = 4
    k: int = 8
    d_x: int = 16
    batch_size: int = 256
    steps: int = 200
    anisotropy_log_span: float = 5.0
    seed: int = 0


@dataclass(frozen=True)
class TtfLdsConfig:
    """Minimal 2-state LDS for Proposition 3.1 demos."""

    alpha: float = 0.5
    epsilon: float = 0.05
    theta: float = 0.2
    horizon: int = 50


@dataclass(frozen=True)
class BenchmarkAnchors:
    """Reported paper anchors for metric stubs (not re-trained here)."""

    humanoid_adamw_median: float = 5200.0
    humanoid_dopr_adamw_median: float = 6100.0
    gsm8k_adamw_peak: float = 78.0
    gsm8k_dopr_adamw_peak: float = 80.5
    tool_hang_adamw_best: float = 0.62
    tool_hang_dopr_adamw_best: float = 0.71
    transport_adamw_best: float = 0.78
    transport_dopr_adamw_best: float = 0.85
