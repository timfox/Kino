"""VOICEGIRAFFE benchmark config (arXiv:2605.27976)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskTier(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class SingleHopTask(str, Enum):
    TEMPORAL_LOCALIZATION = "temporal_localization"
    SEMANTIC_CONTENT = "semantic_content"
    ACOUSTIC_EVENT = "acoustic_event"
    PARALINGUISTIC = "paralinguistic"


class MultiHopTask(str, Enum):
    CAUSAL_ALIGNMENT = "causal_alignment"
    EVENT_TRACKING = "event_tracking"


class InferenceMode(str, Enum):
    E2E = "end_to_end"
    CASCADE = "cascaded_caption"
    LRM = "reasoning_enhanced_cascade"


class Domain(str, Enum):
    SPORTS = "sports"
    ESPORTS = "esports"
    TV_DRAMA = "tv_drama"
    NEWS = "news"
    PODCAST = "podcast"


@dataclass
class VoiceGiraffeConfig:
    paper_arxiv: str = "arXiv:2605.27976"
    org: str = "Future Living Lab, Alibaba"
    hub_repo_id: str = "FutureLivingLab/VOICEGIRAFFE"
    hub_revision: str = "main"
    hub_license: str = "cc-by-nc-4.0"
    n_recordings: int = 123
    total_hours: float = 113.1
    avg_duration_min: float = 55.2
    pct_over_one_hour: float = 0.34
    n_qa_total: int = 1500
    n_single_hop: int = 1000
    n_multi_hop: int = 500
    n_sub_tasks: int = 6
    n_domains: int = 5
    languages: tuple[str, ...] = ("EN", "ZH")
    clip_segment_s: tuple[float, float] = (30.0, 40.0)
    cascade_window_s: float = 30.0
    passing_score_pct: float = 70.0
    # Table 2 anchors
    human_overall_pct: float = 70.51
    human_causal_pct: float = 52.24
    human_event_tracking_pct: float = 61.11
    best_e2e_model: str = "Qwen3.5-Omni-Plus"
    best_e2e_overall_pct: float = 76.00
    best_opensource_cascade_pct: float = 50.60
    best_opensource_cascade_model: str = "Qwen3-Omni"
    # LRM gains (Sec. 4.1)
    opensource_cascade_avg_pct: float = 37.15
    opensource_lrm_avg_pct: float = 55.39
