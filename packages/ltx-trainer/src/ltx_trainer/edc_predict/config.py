"""Config for multi-band EDC prediction stub (arXiv:2605.20968)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdcPredictConfig:
    paper_arxiv: str = "arXiv:2605.20968"
    title: str = "From Numbers to Perception: Energy Decay Curves Prediction"
    repo: str = "https://github.com/TUIlmenauAMS/LSTM-Model-Energy-Decay-Curves"

    n_input_features: int = 16
    n_third_octave_bands: int = 24
    n_rooms: int = 6000

    lstm_params_m: float = 90.0
    convnet_params_m: float = 9.0

    loss_alpha: float = 0.2
    slope_stride_k: int = 50
    rss_stickiness_p: float = 0.9

    t30_jnd_fraction: float = 0.05
