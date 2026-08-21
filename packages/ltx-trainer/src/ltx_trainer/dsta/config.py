"""Configuration for DSTA / Fine-Badminton TAL (arXiv:2605.23355)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSTAConfig:
    paper_arxiv: str = "arXiv:2605.23355"
    zenodo_doi: str = "https://doi.org/10.5281/zenodo.20292976"
    channel_split_alpha: float = 0.5
    input_frames: int = 768
    input_stride: int = 4
    spatial_size: int = 160
    fps: int = 25
    rally_gap_frames: int = 150
    shuttleset_stroke_half_window: int = 9
    tiou_thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    backbone: str = "VideoMAE-B"
    detection_head: str = "ActionFormer"
    base_framework: str = "AdaTAD"


@dataclass
class FineBadmintonStats:
    matches: int = 31
    action_categories: int = 29
    rallies: int = 2104
    annotated_actions: int = 27597
    total_frames: int = 2948689
    zenodo_subset_videos: int = 10
    median_action_duration_frames: int = 20
