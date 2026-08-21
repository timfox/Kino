"""MUSTBENCH: temporal grounding in music LALMs (arXiv:2605.29300)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MustTask(str, Enum):
    TSG = "temporal_source_grounding"
    LTR = "local_transition_recognition"
    TAD = "transition_aware_description"
    GTO = "global_temporal_ordering"
    MTR = "mood_trajectory_reasoning"


@dataclass
class MustBenchConfig:
    paper_arxiv: str = "arXiv:2605.29300"
    backbone: str = "Qwen2.5-Omni"
    must_encoder: str = "MERT+LoRA"
    # Table 1 benchmark stats
    total_qa_pairs: int = 1_264
    unique_songs: int = 517
    avg_duration: str = "3m 42s"
    # Table 2 training split totals
    train_qa_total: int = 40_000
    val_qa_total: int = 4_567
    test_qa_total: int = 1_264
    # Task test counts
    tsg_n: int = 400
    ltr_n: int = 208
    tad_n: int = 208
    gto_n: int = 198
    mtr_n: int = 250
    # Table 3 anchors — MUST 7B
    must_7b_total: float = 44.1
    must_7b_tsg_onset_hit3: float = 55.5
    must_7b_tsg_offset_hit3: float = 62.5
    must_7b_ltr_acc: float = 60.6
    must_7b_gto_acc: float = 67.2
    must_7b_mtr_iou: float = 22.6
    qwen25_7b_total: float = 24.3
    gemini_25_flash_total: float = 41.8
    # MUST training
    sample_rate_hz: float = 24000.0
    must_token_rate: float = 6.66
    must_token_dim: int = 768
    lora_rank: int = 64
    grpo_tsg_scale_s: float = 15.0
    hit_tolerance_s: float = 3.0
