"""ChildVox: child-centered audio benchmark (arXiv:2605.29257)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChildVoxCategory(str, Enum):
    PHYSIOLOGICAL = "physiological"
    VOCALIZATION = "vocalization"
    CANONICAL_SYLLABLES = "canonical_syllables"
    SPEECH = "speech"


@dataclass
class ChildVoxConfig:
    paper_arxiv: str = "arXiv:2605.29257"
    sample_rate_hz: int = 16000
    min_duration_ms: int = 200
    max_duration_s: float = 10.0
    num_datasets: int = 17
    num_subtasks: int = 20
    balanced_total_samples: int = 64_641
    balanced_subtasks: int = 14
    lora_rank: int = 64
    train_per_label: int = 2000
    test_per_label: int = 50
    myst_train_asr: int = 10_000
    myst_test_asr: int = 500
    # Table 4 anchors
    unified_macro_f1_tasks: float = 0.9271  # reported as accuracy in paper; macro-F1 on classification
    nls_der_whisper_large: float = 17.70
    ados_wer_whisper_large: float = 40.20
    myst_wer_whisper_large: float = 14.80
