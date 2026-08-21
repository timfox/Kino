"""PINNS pedestrian–vehicle interaction dataset (Peng et al., arXiv:2605.25947)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PINNSConfig:
    """Defaults from paper Sec. III–IV."""

    resolution: tuple[int, int] = (1536, 864)
    fps: int = 30
    num_scenes: int = 4
    raw_video_hours: float = 100.0  # >6000 minutes

    num_pedestrians: int = 457
    num_vehicles: int = 189
    num_objects: int = 646
    trajectory_points_ped: int = 180344
    trajectory_points_veh: int = 47555

    homography_points_per_scene: tuple[int, int] = (10, 15)
    reprojection_error_px: float = 8.24
    reconstruction_error_m: float = 0.28

    # Benchmark protocol (Sec. IV)
    eval_hz: float = 2.5
    t_obs: int = 8
    t_pred: int = 12
    baseline_model: str = "Trajectron++"

    scenes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Wyoming_USA_crossroad",
            "Tokyo_Japan_crossroad",
            "Pistoia_Italy_park",
            "SuratThani_Thailand_street",
        )
    )

    repo: str = "https://github.com/Songan-Lab"
