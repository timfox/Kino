"""MTAVG-Bench 2.0: cinematic expressiveness failure diagnosis for multi-talker A-V (arXiv:2605.28035)."""

from ltx_trainer.mtavg2.config import MTAVG2Config
from ltx_trainer.mtavg2.diagnosis import (
    attach_mtavg2_to_preprocess_meta,
    diagnose_clip_proxy,
    ltx_script_prompt_suffix,
    merge_preprocess_extra,
    mtavg2_diag_enabled,
    mtavg2_preprocess_extra,
    score_generation_vs_bench,
    write_diagnosis_report,
)
from ltx_trainer.mtavg2.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.mtavg2.scoring import (
    failure_rate,
    holistic_quality_index,
    primary_issue_accuracy,
    rationale_consistency_pct,
    score_multiple_choice,
    score_single_choice,
    temporal_localization_accuracy,
    weighted_average_sub_dims,
)
from ltx_trainer.mtavg2.taxonomy import (
    CATEGORIES,
    FAILURE_MODES,
    SUB_DIMENSION_BY_CODE,
    failure_modes_for_sub_dim,
    taxonomy_card,
)

__all__ = [
    "CATEGORIES",
    "FAILURE_MODES",
    "MTAVG2Config",
    "SUB_DIMENSION_BY_CODE",
    "attach_mtavg2_to_preprocess_meta",
    "diagnose_clip_proxy",
    "evaluation_demo",
    "evaluation_smoke",
    "failure_modes_for_sub_dim",
    "failure_rate",
    "framework_card",
    "holistic_quality_index",
    "knowledge_card",
    "ltx_script_prompt_suffix",
    "merge_preprocess_extra",
    "mtavg2_diag_enabled",
    "mtavg2_preprocess_extra",
    "primary_issue_accuracy",
    "rationale_consistency_pct",
    "score_generation_vs_bench",
    "score_multiple_choice",
    "score_single_choice",
    "taxonomy_card",
    "temporal_localization_accuracy",
    "weighted_average_sub_dims",
    "write_diagnosis_report",
]
