"""Configuration for PINN synchronization control (arXiv:2601.00178)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyncPinnConfig:
    paper_arxiv: str = "2601.00178"
    paper_title: str = (
        "Controlling synchronization dynamics via physics-informed neural networks"
    )
    author: str = "Kaiming Luo (Fudan University)"

    # Baseline Kuramoto experiment (Sec. IV A)
    n_oscillators: int = 10
    coupling_k: float = 0.05
    fully_connected: bool = True
    r_target: float = 1.0
    t_target: float = 2.0
    control_horizon: float = 5.0

    # Baseline feedback gains (Sec. IV C)
    phase_feedback_gain: float = 1.5
    freq_comp_gain: float = 1.0

    # Kuramoto–Sakaguchi frustrated regime (Sec. IV C)
    sakaguchi_alpha: float = 1.5707963267948966  # pi/2
    sakaguchi_k: float = 0.4

    # Noise robustness sweep (Sec. IV D)
    noise_sigma_max: float = 0.5
