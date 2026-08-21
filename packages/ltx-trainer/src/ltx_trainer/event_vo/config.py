"""Configuration for event-based monocular ESKF odometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventVOConfig:
    paper_arxiv: str = "2605.27661"
    paper_title: str = (
        "Design of a Real-time Asynchronous Monocular Odometry for Planetary Exploration"
    )
    authors: str = "Iñigo, Steidle, Stürzl (DLR / University of Zaragoza)"

    # Intrinsics (undistorted, LUT from RATE)
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 320.0
    cy: float = 240.0
    image_w: int = 640
    image_h: int = 480

    # ESKF noise (paper Sec. IV-A: σa,σw=2.0, σu=σv=1.0 px)
    process_noise_pos: float = 2.0
    process_noise_vel: float = 2.0
    process_noise_rot: float = 2.0
    measurement_noise_px: float = 1.0

    # Feature management
    min_parallax_px: float = 8.0
    homography_init_features: int = 40
    max_landmarks: int = 200

    # Reference results (paper Sec. IV)
    sim_mape_m: float = 0.065
    uzh_poster_mape_m: float = 0.06
    uzh_poster_rms_m: float = 0.06
