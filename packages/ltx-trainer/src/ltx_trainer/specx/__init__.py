"""SpecX — multimodal spectroscopy benchmark (arXiv:2605.18791)."""

from ltx_trainer.specx.config import SpecxConfig
from ltx_trainer.specx.layout import LIMITATIONS, TASK_DESCRIPTIONS, TIER_DESCRIPTIONS
from ltx_trainer.specx.metrics import cosine_similarity, macro_f1_multilabel, top_k_accuracy
from ltx_trainer.specx.mock import evaluation_smoke, pipeline_demo
from ltx_trainer.specx.tokenize import discretize_ir_spectrum
from ltx_trainer.specx.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
)
from ltx_trainer.specx.tables import (
    table1_benchmark_comparison,
    table3_elucidation_random_excerpt,
    table4_functional_group_random,
    table11_subset_modalities,
    modality_representations,
)

__all__ = [
    "LIMITATIONS",
    "TASK_DESCRIPTIONS",
    "TIER_DESCRIPTIONS",
    "SpecxConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "cosine_similarity",
    "discretize_ir_spectrum",
    "evaluation_smoke",
    "framework_card",
    "macro_f1_multilabel",
    "pipeline_demo",
    "top_k_accuracy",
    "headline_results",
    "modality_representations",
    "table1_benchmark_comparison",
    "table3_elucidation_random_excerpt",
    "table4_functional_group_random",
    "table11_subset_modalities",
]
