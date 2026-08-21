"""SVHighlights + TF-SELECTOR configuration (arXiv:2606.06926, KDD 2026)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SportCategory(str, Enum):
    AMERICAN_FOOTBALL = "american_football"
    BASEBALL = "baseball"
    BASKETBALL = "basketball"
    ICE_HOCKEY = "ice_hockey"
    RACING = "racing"
    RUGBY = "rugby"
    SOCCER = "soccer"
    VOLLEYBALL = "volleyball"


@dataclass
class SvHighlightsConfig:
    paper_arxiv: str = "2606.06926"
    paper_doi: str = "10.1145/3770855.3817564"
    paper_url: str = "https://leedongkyu2019.github.io/SVHighlights/"

    n_videos: int = 320
    total_hours: float = 640.18
    avg_duration_min: float = 120.0
    videos_per_sport: int = 40
    sports: tuple[SportCategory, ...] = field(default_factory=lambda: tuple(SportCategory))

    clip_duration_s: float = 2.0
    highlight_window_s: float = 1.0
    label_overlap_threshold: float = 0.5

    alignment_downsample_p: int = 144
    alignment_tau: float = 5.0
    psnr_filter_threshold: float = 20.0
    highlight_sample_interval_s: float = 1.0

    max_segment_length_s: float = 120.0
    transcript_gap_merge_s: float = 1.0
    saliency_min: float = 0.0
    saliency_max: float = 5.0

    shot_detector: str = "TransNetV2"
    asr_model: str = "WhisperX-large-v2"
    vlm_model: str = "InternVL2.5-8B"
    llm_model: str = "Llama-3-8B"

    vtg_query_template: str = "Highlight of this {video_type} video"
    random_seed: int = 42
