"""IG temporal sound event detection stub (arXiv:2605.23293)."""

from __future__ import annotations

from dataclasses import dataclass, field


DOMESTIC_CLASSES: tuple[str, ...] = (
    "Alarm bell ringing",
    "Blender",
    "Cat",
    "Dishes",
    "Dog",
    "Electric shaver/toothbrush",
    "Frying",
    "Running water",
    "Speech",
    "Vacuum cleaner",
)


@dataclass
class IgSedConfig:
    paper_arxiv: str = "arXiv:2605.23293"
    title: str = "IG temporal detection on clip-level sound classifiers"
    sample_rate_hz: int = 32_000
    clip_duration_s: float = 10.0
    num_classes: int = 10
    ig_steps: int = 50
    frame_resolution_ms: int = 100
    framewise_frames: int = 31
    train_clips: int = 823
    val_clips: int = 96
    test_clips: int = 97
    clip_threshold: float = 0.5
    ig_peak_iou_percentile: int = 56
    ig_peak_f1_percentile: int = 57
    mean_iou_ig: float = 0.39
    mean_f1_ig: float = 0.52
    pointing_game_ig: float = 0.826
    mean_iou_fw_ws: float = 0.42
    mean_f1_fw_ws: float = 0.55
    pointing_game_fw_ws: float = 0.973
    mean_iou_fw_ss: float = 0.45
    mean_f1_fw_ss: float = 0.58
    pointing_game_fw_ss: float = 0.979
    classes: tuple[str, ...] = field(default_factory=lambda: DOMESTIC_CLASSES)
