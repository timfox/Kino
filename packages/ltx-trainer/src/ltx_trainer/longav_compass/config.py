"""LongAV-Compass: minute-scale audio-visual generation benchmark (arXiv:2605.26224)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class X2AVTask(str, Enum):
    T2AV = "T2AV"
    I2AV = "I2AV"
    V2AV = "V2AV"


class ApplicationScenario(str, Enum):
    VLOG = "Vlog"
    CONTENT_CREATOR = "Content-Creator"
    PERFORMANCE_ADS = "Performance Ads"
    BRAND_ADS = "Brand Ads"


class ComplexityLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


@dataclass
class LongAVCompassConfig:
    paper_arxiv: str = "arXiv:2605.26244"  # cs.CV minute-scale X2AV benchmark
    github: str = "https://github.com/pkucs-Ltf/LongAV-Compass"
    judge_model: str = "Gemini 3.1 Pro"
    target_duration_s_min: int = 60
    target_duration_s_max: int = 120
    n_samples: int = 284
    n_t2av: int = 128
    n_i2av: int = 115
    n_v2av: int = 41
    n_events_t2av: int = 879
    n_shots_t2av: int = 2115
    n_events_i2av: int = 807
    n_shots_i2av: int = 1989
    n_events_v2av: int = 235
    n_shots_v2av: int = 731
    n_evaluated_models: int = 11
    human_alignment_pearson: tuple[float, float, float] = (0.917, 0.935, 0.867)  # content, visual, stability
