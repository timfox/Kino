"""CPC3 frame-aligned Canary–WavLM fusion (arXiv:2605.23619)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cpc3FafConfig:
    paper_arxiv: str = "arXiv:2605.23619"
    challenge: str = "3rd Clarity Prediction Challenge (CPC3)"
    canary_model: str = "nvidia/canary-1b-flash"
    wavlm_model: str = "microsoft/wavlm-large"
    canary_layers: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17)
    wavlm_layers_main: tuple[int, ...] = (17, 18, 19, 20, 21, 22, 23, 24)
    canary_hz: float = 12.5
    wavlm_hz: float = 50.0
    downsample_stride: int = 4
    hidden_single: int = 256
    hidden_dual: int = 192
    learning_rate: float = 1e-4
    batch_size: int = 64
    best_eval_rmse: float = 24.96
    best_eval_corr: float = 0.796
    canary_only_eval_rmse: float = 25.64
    score_avg_eval_rmse: float = 25.53
