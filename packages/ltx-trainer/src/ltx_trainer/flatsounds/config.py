"""Configuration for FlatSounds V2A physical benchmark (arXiv:2605.30339)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlatSoundsConfig:
    paper_arxiv: str = "arXiv:2605.30339"
    dataset_clips: int = 185
    physics_pairs: int = 178
    physics_singles: int = 90
    physics_test_cases: int = 268
    num_seeds: int = 10
    sample_rate_hz: int = 16000
    analysis_sr_hz: int = 16000
    hit_tolerance_ms_min: float = 100.0
    hit_tolerance_ms_max: float = 250.0
    effect_size_frac: float = 0.02
    effect_size_mad_mult: float = 0.25
    fold_role: str = "v2a_physical_benchmark_proxy"
