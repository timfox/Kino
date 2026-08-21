"""MTLLFM configuration (arXiv:2605.25409)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MTLLFMConfig:
    """Reference hyperparameters from MTLLFM."""

    paper_arxiv: str = "arXiv:2605.25409"
    github: str = "https://github.com/WSCSports/MTLLFM-temporal-laughter-localization"
    hidden_dim: int = 1024
    audio_dim: int = 1024
    visual_dim: int = 768
    segment_seconds: float = 5.0
    localization_bins: int = 10
    sharpen_temperature: float = 0.5
    hubert_hz: float = 50.0
    mae_fps: float = 10.0
