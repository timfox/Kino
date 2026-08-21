"""Ethical pluralism: normative simplex + stacked ensemble (arXiv:2605.28707)."""

from ltx_trainer.ethical_pluralism.benchmark import (
    benchmark_card,
    generate_benchmark,
    train_test_split,
)
from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.embeddings import triple_bert_supervector
from ltx_trainer.ethical_pluralism.ensemble import (
    StackedEnsembleModel,
    ablation_study,
    build_feature_matrix,
    evaluate_classifier,
    stratified_cross_validate,
    train_stacked_ensemble,
    transformer_ablation,
)
from ltx_trainer.ethical_pluralism.features import EthicalCase, case_feature_vector, infer_normative_scores
from ltx_trainer.ethical_pluralism.mock import evaluation_smoke
from ltx_trainer.ethical_pluralism.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.ethical_pluralism.pluralism import pluralism_report, temperature_scale_probs
from ltx_trainer.ethical_pluralism.simplex import (
    plurality_features,
    project_simplex,
    simplex_constraint_ok,
)
from ltx_trainer.ethical_pluralism.taxonomy import (
    BENCHMARK_SIZE,
    SUBTHEORIES,
    SUBTHEORY_BY_ID,
    SUBTHEORY_IDS,
)

__all__ = [
    "BENCHMARK_SIZE",
    "EthicalCase",
    "EthicalPluralismConfig",
    "SUBTHEORIES",
    "SUBTHEORY_BY_ID",
    "SUBTHEORY_IDS",
    "StackedEnsembleModel",
    "ablation_study",
    "build_feature_matrix",
    "benchmark_card",
    "benchmarks_bundle",
    "case_feature_vector",
    "evaluate_classifier",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "generate_benchmark",
    "infer_normative_scores",
    "knowledge_card",
    "plurality_features",
    "pluralism_report",
    "project_simplex",
    "simplex_constraint_ok",
    "temperature_scale_probs",
    "stratified_cross_validate",
    "train_stacked_ensemble",
    "train_test_split",
    "transformer_ablation",
    "triple_bert_supervector",
]
