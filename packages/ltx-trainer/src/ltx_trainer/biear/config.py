"""BiEAR — adaptive binaural front-end (arXiv:2606.06795)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BiearConfig:
    paper_arxiv: str = "arXiv:2606.06795"
    title: str = (
        "BiEAR: A Human Auditory-Inspired Adaptive Binaural Front-end for "
        "Multi-Speaker Localisation and Distance Estimation"
    )
    framework: str = "BiEAR"
    github: str = "https://github.com/Hanyu-Meng/BiEAR"

    # Task (§2.1)
    n_sectors: int = 8
    sector_deg: float = 45.0
    segment_s: float = 1.0

    # STFT / filterbank (§3.2)
    sample_rate_hz: int = 16000
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    n_subbands: int = 100
    q_min: float = 0.05
    q_max: float = 30.0
    ema_beta: float = 0.8

    # Q control (§2.3.3)
    abs_delta_q_base: float = 2.0
    abs_m_low: float = 0.5
    abs_m_high: float = 5.0
    rel_delta_q_base: float = 1.0
    rel_m_low: float = 0.3
    rel_m_high: float = 5.0

    # Training (§3.2)
    lambda_detect: float = 0.25
    lambda_azimuth: float = 0.45
    lambda_distance: float = 0.35
    batch_size: int = 64
    learning_rate: float = 1e-4

    # Table 2 — BiEAR + Dual Controller + Rel (1-speaker seen / unseen)
    azim_mae_1spk_seen: float = 0.36
    azim_mae_1spk_unseen: float = 0.39
    dist_acc_1spk_seen: float = 97.84
    detect_acc_3spk_seen: float = 90.82

    # Table 3 — meeting room + env transfer (1-speaker)
    meeting_detect_transfer: float = 93.74
    meeting_azim_mae_transfer: float = 3.92
    meeting_dist_transfer: float = 93.98

    # Baseline DeepEar anechoic 1-spk azim MAE (seen)
    deepear_azim_mae_1spk: float = 0.80
