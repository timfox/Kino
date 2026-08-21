"""Before the Shutter configuration (arXiv:2605.30318)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeforeShutterConfig:
    paper_arxiv: str = "arXiv:2605.30318"
    repo_url: str = "https://github.com/songrise/Before-the-Shutter"
    iso_fixed: int = 100
    exposure_stops: float = 3.0  # s- and s+ in Eq. (8)
    max_staging_steps: int = 3
    max_composition_steps: int = 7
    max_lighting_steps: int = 7
    frontier_size: int = 6
    benchmark_tasks: int = 50
    benchmark_scenes: int = 14
    benchmark_indoor: int = 8
    benchmark_outdoor: int = 6
    contact_threshold_m: float = 0.02
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    default_f_number: float = 2.8
    default_focal_mm: float = 50.0
    lighting_presets: tuple[str, ...] = (
        "rembrandt",
        "split",
        "butterfly",
        "loop",
        "rim_key_fill",
        "chiaroscuro",
    )


@dataclass
class PortraitPlanState:
    """Single planning state s = (H, C, L) stub."""

    body_pose: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    root_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_extrinsic: tuple[float, float, float] = (1.5, 0.0, 1.6)
    focal_mm: float = 50.0
    f_number: float = 2.8
    exposure_comp_stops: float = 0.0
    light_powers_w: tuple[float, ...] = (300.0, 100.0, 80.0)
    light_distances_m: tuple[float, ...] = (1.2, 1.5, 2.0)
    preset: str = "rembrandt"
    stage: str = "staging"
