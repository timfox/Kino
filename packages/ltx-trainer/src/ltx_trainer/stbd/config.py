"""Subspace TBD for passive multi-target tracking (arXiv:2605.25498)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StbdConfig:
    paper_arxiv: str = "arXiv:2605.25498"
    venue: str = "APSIPA ASC 2026"
    num_sensors: int = 40
    num_target_slots: int = 2
    num_frames: int = 200
    frame_dt_s: float = 0.128
    room_size_m: float = 3.0
    speed_of_sound_m_s: float = 343.0
    num_freq_bins: int = 61
    kappa_per_freq: float = 10.0
    process_noise_q: float = 0.09
    birth_velocity_std_m_s: float = 0.5
    boundary_tau_m: float = 0.05
    particle_counts: tuple[int, ...] = (2000, 4000, 8000)
    snr_db_eval: tuple[float, ...] = (-10.0, 0.0, 10.0)
    activity_switch_frame: int = 100
