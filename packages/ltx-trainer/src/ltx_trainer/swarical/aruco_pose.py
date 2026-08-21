"""ArUco + Raspberry Pi Camera Module 3 pose estimation (companion Sec. 3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CameraLens(str, Enum):
    REGULAR = "regular"
    WIDE = "wide"


class RangeZone(str, Enum):
    """Fig. 6 — blind, sweet, and decaying detection ranges."""

    BLIND = "blind"
    SWEET = "sweet"
    DECAYING = "decaying"
    OUT_OF_RANGE = "out_of_range"


@dataclass
class ArucoReproConfig:
    """Defaults from companion Sec. 3 and aruco-pose-estimation README."""

    companion_doi: str = "10.1145/3746027.3759199"
    aruco_repo: str = "https://github.com/flyinglightspeck/aruco-pose-estimation"
    marker_size_m: float = 0.0047
    frame_resolution: str = "720p"
    camera_interface: str = "pi3"
    live_duration_s: int = 10
    wide_blind_mm: float = 50.0
    sweet_range_mm: tuple[float, float] = (60.0, 80.0)
    orientation_error_deg_max: float = 1.0


def classify_detection_range(
    distance_mm: float,
    *,
    lens: CameraLens = CameraLens.WIDE,
    cfg: ArucoReproConfig | None = None,
) -> RangeZone:
    """Classify distance into blind / sweet / decaying (companion Figs. 6–7)."""
    cfg = cfg or ArucoReproConfig()
    if lens == CameraLens.WIDE and distance_mm < cfg.wide_blind_mm:
        return RangeZone.BLIND
    lo, hi = cfg.sweet_range_mm
    if lo <= distance_mm <= hi:
        return RangeZone.SWEET
    if distance_mm > hi:
        return RangeZone.DECAYING
    if distance_mm < lo:
        return RangeZone.BLIND if lens == CameraLens.WIDE else RangeZone.DECAYING
    return RangeZone.OUT_OF_RANGE


def percent_distance_error(
    distance_mm: float,
    *,
    lens: CameraLens = CameraLens.WIDE,
    display: str = "paper",
) -> float | None:
    """Synthetic error curve aligned with paper Figs. 6–9 (sweet ≈ 1.15% at 6–8 cm)."""
    zone = classify_detection_range(distance_mm, lens=lens)
    if zone == RangeZone.BLIND:
        return None
    lo, hi = (60.0, 80.0)
    if zone == RangeZone.SWEET:
        base = 1.15
        if display == "lcd":
            base *= 1.08
        return base
    # Decaying: error grows with distance past sweet range.
    if distance_mm > hi:
        excess = (distance_mm - hi) / 100.0
        base = 1.15 + 0.35 * excess
        return base * (1.05 if lens == CameraLens.WIDE else 1.0)
    mid = (lo + distance_mm) / 2.0
    return 1.5 + 0.02 * abs(mid - 70.0)


def pi_pose_estimation_command(
  cfg: ArucoReproConfig | None = None,
  *,
  live: bool = True,
  save: bool = False,
  expected_distance: str = "80mm",
) -> str:
    """Shell command from companion Sec. 3 (run on Raspberry Pi 5)."""
    cfg = cfg or ArucoReproConfig()
    parts = [
        "python pi_pose_estimation.py",
        f"-i {cfg.camera_interface}",
        f"-r {cfg.frame_resolution}",
        f"-t {cfg.live_duration_s}",
        f"--marker_size {cfg.marker_size_m}",
    ]
    if live:
        parts.append("--live")
    if save:
        parts.extend(["--save", f"-e {expected_distance}"])
    return " ".join(parts)


def aruco_reproduction_guide() -> dict[str, Any]:
    """Hardware + software steps for pose-estimation companion (Sec. 3–4)."""
    cfg = ArucoReproConfig()
    return {
        "companion_section": "Position Estimation Using a Camera",
        "repository": cfg.aruco_repo,
        "hardware_paper": [
            "Raspberry Pi 5 (bookworm)",
            "Raspberry Pi Camera Module 3 NoIR (wide or regular)",
            "Printed ArUco marker on paper (4.7 mm) or Waveshare 1.3\" LCD + Arduino Uno",
            "3D printed holder (assets/paper or assets/lcd)",
        ],
        "setup_commands": [
            "git clone https://github.com/flyinglightspeck/aruco-pose-estimation.git",
            "cd aruco-pose-estimation && bash setup.sh",
            "rpicam-hello --list-cameras",
            "source .env/bin/activate",
        ],
        "run_live": pi_pose_estimation_command(cfg, live=True, save=False),
        "run_save_batch": pi_pose_estimation_command(cfg, live=False, save=True, expected_distance="80mm"),
        "post_process": "python process.py  # batch /results for cameraplots.nb",
        "zones": {
            "wide_blind_mm": f"< {cfg.wide_blind_mm}",
            "sweet_mm": list(cfg.sweet_range_mm),
            "decaying": "beyond sweet; error increases with distance",
        },
        "swazure_note": "FLSs closer than 50 mm (wide blind range) cooperate via Swazure [5] in full Swarical stack.",
        "reproducibility_study": {
            "hardware_experiments": "4 of 5 reproduced; wide+NoIR vs paper regular lens",
            "video_documentation": "Reviewer-requested setup videos incorporated in companion",
        },
    }


def sample_distance_error_table(
    *,
    lens: CameraLens = CameraLens.WIDE,
    distances_mm: tuple[float, ...] = (40.0, 60.0, 70.0, 80.0, 100.0, 150.0),
) -> list[dict[str, Any]]:
    """Tabular blind/sweet/decaying + percent error (Figs. 6–8 companion)."""
    rows: list[dict[str, Any]] = []
    for d in distances_mm:
        zone = classify_detection_range(d, lens=lens)
        rows.append(
            {
                "distance_mm": d,
                "lens": lens.value,
                "zone": zone.value,
                "pct_error": percent_distance_error(d, lens=lens),
            }
        )
    return rows
