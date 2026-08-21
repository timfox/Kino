"""GrowLoop: self-evolving conversation evaluation (Amap Voice, arXiv:2605.28882)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvaluationZone(str, Enum):
    CONSENSUS = "consensus"
    DIVERGENCE = "divergence"


class RubricTrack(str, Enum):
    SAFETY = "safety"
    QUALITY = "quality"


class GrowStage(str, Enum):
    COLD_START = "cold_start"
    HEURISTIC_LEARNING = "heuristic_learning"
    CASCADED_INTEGRATION = "cascaded_integration"


@dataclass
class GrowLoopConfig:
    paper_arxiv: str = "arXiv:2605.28882"
    org: str = "Amap Voice, Alibaba Group"
    judge_llm: str = "Gemini 3.1 Pro Preview"
    # Annotator agreement (§3.2)
    inter_annotator_agreement: float = 0.563
    # Rubric setup (§4.1)
    seed_cases: int = 50
    seed_responses: int = 200
    quality_dimensions: int = 18
    cognitive_categories: int = 4
    safety_convergence_target: float = 0.90
    quality_convergence_target: float = 0.85
    safety_final_agreement: float = 0.915
    quality_initial_agreement: float = 0.654
    quality_final_agreement: float = 0.866
    safety_iterations: int = 6
    quality_iterations: int = 10
    merged_agreement_gemini: float = 0.860
    merged_agreement_claude: float = 0.836
    # Case setup
    case_count: int = 500
    real_conversations: int = 1_767
    real_user_messages: int = 12_799
    csp_fields: int = 15
    trap_categories: int = 10
    # Hard gates (Table 5)
    gate_diversity_min: float = 55.0
    gate_kendall_tau_min: float = 0.70
    gate_cliffs_delta_min: float = 0.32
    gate_adjacent_gap_min: float = 5.0
    gate_best_mean_lo: float = 60.0
    gate_best_mean_hi: float = 75.0
    # Realized gate metrics (§4.3)
    diversity_score: float = 72.3
    kendall_tau_mean: float = 0.713
    cliffs_delta_min: float = 0.33
    best_tier_mean: float = 69.5
    hnorm_domain: float = 0.941
    # Table 8 GrowLoop vs baselines
    growloop_tie_aware_acc: float = 0.78
    growloop_pair_acc: float = 0.87
    growloop_spearman: float = 0.78
    icai_tie_aware_acc: float = 0.58
    # Tier means (Table 11)
    tier_means: tuple[float, float, float, float] = (69.5, 58.1, 46.5, 22.6)
    tier_fatal_pct: tuple[float, float, float, float] = (12.8, 23.4, 34.9, 64.7)
    # Heuristic learning
    rubric_length_cap_chars: int = 24_000
    must_token_rate: float = 6.66
