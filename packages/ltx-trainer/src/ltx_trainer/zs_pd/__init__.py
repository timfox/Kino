"""Zero-shot Parkinson's disease detection from speech (LLM vs LALM)."""

from ltx_trainer.zs_pd.aggregation import subject_label_from_segments
from ltx_trainer.zs_pd.config import ZsPdConfig
from ltx_trainer.zs_pd.datasets import table_i_datasets
from ltx_trainer.zs_pd.features import (
    FEATURE_NAMES,
    serialize_features_for_llm,
    toy_feature_vector,
)
from ltx_trainer.zs_pd.layout import LIMITATIONS
from ltx_trainer.zs_pd.metrics import (
    auroc_rank,
    balanced_accuracy,
    brier_score,
    sensitivity_specificity,
)
from ltx_trainer.zs_pd.mock import evaluation_smoke
from ltx_trainer.zs_pd.pipeline import (
    benchmarks_bundle,
    best_per_dataset,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_iii_results,
)
from ltx_trainer.zs_pd.prompts import LALM_PROMPT_TEMPLATE, LLM_PROMPT_TEMPLATE, build_llm_prompt

__all__ = [
    "FEATURE_NAMES",
    "LALM_PROMPT_TEMPLATE",
    "LIMITATIONS",
    "LLM_PROMPT_TEMPLATE",
    "ZsPdConfig",
    "auroc_rank",
    "balanced_accuracy",
    "benchmarks_bundle",
    "best_per_dataset",
    "brier_score",
    "build_llm_prompt",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "sensitivity_specificity",
    "serialize_features_for_llm",
    "subject_label_from_segments",
    "table_i_datasets",
    "table_iii_results",
    "toy_feature_vector",
]
