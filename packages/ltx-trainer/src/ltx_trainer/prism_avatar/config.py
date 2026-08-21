"""PrismAvatar configuration (Fang et al. arXiv:2606.10550)."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2606.10550"
PAPER_TITLE = (
    "PrismAvatar: Pseudo-Multiview Reconstruction and Subpixel Prism Rendering "
    "for Real-Time Stereoscopic Communication"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

YAW_BINS = (15.0, 20.0, 25.0)
YAW_ACCEPT_MIN = 14.0
YAW_ACCEPT_MAX = 27.0
YAW_BIN_TOLERANCE = 2.5

DISPLAY_VIEWS = 32
DISPLAY_VIEW_MIN = -25.0
DISPLAY_VIEW_MAX = 25.0
VIEW_RES = (960, 540)
PANEL_RES = (3840, 2160)

LIVE_FPS = 10.65
STUDENT_FPS = 38.49


@dataclass
class PrismAvatarConfig:
    """Monocular Gaussian head avatar + lenticular subpixel routing."""

    yaw_bins: tuple[float, ...] = YAW_BINS
    num_views: int = DISPLAY_VIEWS
    view_height: int = VIEW_RES[1]
    view_width: int = VIEW_RES[0]
    panel_height: int = PANEL_RES[1]
    panel_width: int = PANEL_RES[0]
    pmv_start_iter: int = 320
    pmv_ramp_iters: int = 700
    pmv_gamma_cap: float = 0.65
    align_iou_min: float = 0.32
    align_iou_edge_min: float = 0.12
    align_centroid_max_px: float = 28.0
    hh_dilate_px: int = 25
    support_tau_s: float = 0.35
    risk_tau_r: float = 0.25
    base_weights: tuple[float, float, float, float] = (1.0, 0.08, 0.05, 0.35)
    pmv_weights_side: tuple[float, float, float, float, float, float] = (
        0.016,
        0.012,
        0.025,
        0.012,
        0.035,
        0.004,
    )
    pmv_weights_color: tuple[float, float, float, float, float, float] = (
        0.016,
        0.004,
        0.012,
        0.006,
        0.026,
        0.010,
    )
    display_cr: float = 1.0
    display_delta_r: float = 3.0
    display_s_ref: float = 0.0
    display_i_ref: int = 16
    camera_radius: float = 0.45
