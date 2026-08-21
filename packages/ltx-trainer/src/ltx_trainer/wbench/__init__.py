"""WBENCH: multi-turn interactive video world model benchmark (Ying et al., arXiv:2605.25874).

Reference building blocks: unified navigation (text / 6-DoF / discrete), NavScore (Eq. 8–14),
VLM checklist scorers, and paper benchmark tables. Full MegaSaM / VLM / Qwen3-VL pipelines are
**not** bundled — plug external evaluators from https://github.com/meituan-longcat/WBench.
"""

from ltx_trainer.wbench.config import WBENCHConfig
from ltx_trainer.wbench.metrics import (
    batch_adjacent_cosine,
    causal_fidelity_case_score,
    event_editing_turn_score,
    gated_spatial_consistency,
    hpsv3_norm,
    mean_sub_metrics,
    perspective_switching_turn_score,
    rescale_0_100,
    subject_action_turn_score,
    temporal_flickering_score,
    vlm_binary_checklist_score,
)
from ltx_trainer.wbench.navigation import (
    action_to_text,
    arc_length_resample,
    build_orbit_trajectory,
    build_translation_trajectory,
    nav_score,
    nav_score_from_trajectories,
    normalized_ate_rotation,
    normalized_ate_translation,
)
from ltx_trainer.wbench.pipeline import (
    benchmark_comparison,
    dataset_composition,
    evaluation_demo,
    framework_card,
    human_preference_alignment,
    sub_metric_index,
    table_full_text_driven,
    table_navigation_split,
    training_step_demo,
)

__all__ = [
    "WBENCHConfig",
    "action_to_text",
    "arc_length_resample",
    "batch_adjacent_cosine",
    "benchmark_comparison",
    "build_orbit_trajectory",
    "build_translation_trajectory",
    "causal_fidelity_case_score",
    "dataset_composition",
    "evaluation_demo",
    "event_editing_turn_score",
    "framework_card",
    "gated_spatial_consistency",
    "hpsv3_norm",
    "human_preference_alignment",
    "mean_sub_metrics",
    "nav_score",
    "nav_score_from_trajectories",
    "normalized_ate_rotation",
    "normalized_ate_translation",
    "perspective_switching_turn_score",
    "rescale_0_100",
    "sub_metric_index",
    "subject_action_turn_score",
    "table_full_text_driven",
    "table_navigation_split",
    "temporal_flickering_score",
    "training_step_demo",
    "vlm_binary_checklist_score",
]
