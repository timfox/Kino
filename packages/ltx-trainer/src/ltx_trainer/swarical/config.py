"""Configuration for Swarical FLS localization (MM '24, arXiv:2605.23774)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class FlsCameraOrientation(str, Enum):
    """Heterogeneous FLS camera mount (Fig. 3)."""

    TOP = "top"
    SIDE = "side"
    BOTTOM = "bottom"


class LocalizationMode(str, Enum):
    """Online decentralized localization techniques (Sec. 4)."""

    HC = "HC"  # Highly Concurrent
    ISR = "ISR"  # Inter-Swarm Rounds (recommended)
    RSF = "RSF"  # Rounds across Swarm-tree and FLS-trees


@dataclass
class SwaricalConfig:
    paper_arxiv: str = "arXiv:2605.23774"
    acm_doi: str = "10.1145/3664647.3681080"
    github: str = "https://github.com/flyinglightspeck/Swarical"
    companion_arxiv: str = "arXiv:2605.26313"
    companion_doi: str = "10.1145/3746027.3759199"
    aruco_pose_repo: str = "https://github.com/flyinglightspeck/aruco-pose-estimation"
    small_scale_fls: int = 16
    small_scale_grid: tuple[int, int] = (4, 4)
    small_scale_localization_s: int = 60

    # Default planner / evaluation (Skateboard, Sec. 5).
    default_group_size_g: int = 50
    default_dead_reckoning_error_deg: float = 5.0
    tracking_range_cm_min: float = 6.0
    tracking_range_cm_max: float = 8.0
    hausdorff_tolerance_pct: float = 5.0
    fls_sphere_radius_cm: float = 1.0

    # Localization timing (Sec. 4).
    localization_timeout_ms: int = 500
    recommended_mode: LocalizationMode = LocalizationMode.ISR

    # Skateboard point cloud (Sec. 5.2).
    skateboard_fls_count: int = 1372
    skateboard_standby_fls: int = 40
    skateboard_camera_mix_pct: tuple[float, float, float] = (10.1, 80.5, 9.4)  # top, side, bottom

    # Fig. 13 plateau (Skateboard, G=50, ISR).
    skateboard_hd_plateau_mm: float = 18.9
    camera_pct_error_at_range: float = 1.15  # Sec. 5.5 discussion

    # SwarMer comparison (Sec. 5.4).
    swarmer_speedup_factor: float = 2.0
    swarmer_distance_reduction_pct: float = 8.0


@dataclass
class RaspberryCameraSpec:
    """Table 1 — Raspberry Camera Module 3 NoIR."""

    lens: str
    resolution_px: tuple[int, int]
    fov_deg: tuple[float, float, float]  # D, H, V
    min_focus_mm: float
    weight_g: float
    price_usd: float


def camera_specs_table1() -> list[RaspberryCameraSpec]:
    return [
        RaspberryCameraSpec("Regular", (4608, 2592), (75.0, 66.0, 41.0), 100.0, 3.2, 25.0),
        RaspberryCameraSpec("Wide", (4608, 2592), (120.0, 102.0, 67.0), 50.0, 3.2, 35.0),
    ]


@dataclass
class CameraPerformanceRow:
    """Table 2 — 720p used in paper experiments."""

    resolution: str
    lens: str
    fps: float
    avg_camera_delay_ms: float
    avg_processing_ms: float


def camera_performance_table2() -> list[CameraPerformanceRow]:
    return [
        CameraPerformanceRow("720p", "Regular", 46.9, 3.0, 18.0),
        CameraPerformanceRow("720p", "Wide", 44.4, 8.0, 14.0),
        CameraPerformanceRow("480p", "Regular", 59.3, 10.0, 6.0),
        CameraPerformanceRow("480p", "Wide", 44.8, 15.0, 7.0),
        CameraPerformanceRow("1080p", "Regular", 21.1, 8.0, 39.0),
        CameraPerformanceRow("1080p", "Wide", 26.0, 8.0, 29.0),
    ]


def fls_density_per_area(*, t_min: float, t_max: float, r: float) -> tuple[float, float]:
    r"""Min/max FLS density per unit area (Sec. 3.1).

    ``D_min = 1 / (π max(T_max/2, R)²)``, ``D_max = 1 / (π max(T_min/2, R)²)``.
    """
    d_min = 1.0 / (math.pi * max(t_max / 2.0, r) ** 2)
    d_max = 1.0 / (math.pi * max(t_min / 2.0, r) ** 2)
    return d_min, d_max


def swarm_count(f_fls: int, group_size_g: int) -> int:
    """``n_G = ⌈F / G⌉`` (Sec. 3.2)."""
    if group_size_g <= 0:
        raise ValueError("group_size_g must be positive")
    return math.ceil(f_fls / group_size_g)
