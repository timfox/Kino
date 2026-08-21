"""CAPT weakly-supervised phoneme-level pronunciation scoring stub."""

from ltx_trainer.capt_weak_phn.config import CaptWeakPhnConfig
from ltx_trainer.capt_weak_phn.gop import batch_gop_features, gop_feature_vector, gop_from_log_posteriors
from ltx_trainer.capt_weak_phn.layout import LIMITATIONS
from ltx_trainer.capt_weak_phn.mock import evaluation_smoke
from ltx_trainer.capt_weak_phn.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_3_test_pcc,
    framework_card,
    headline_results,
    table_1_dev_results,
)
from ltx_trainer.capt_weak_phn.pooling import attn_pool, attention_weights, mean_pool, pool_over_spans
from ltx_trainer.capt_weak_phn.two_stage import (
    SelectionResult,
    select_balanced_by_bins,
    select_unbalanced,
    simulate_finetune_gain,
)

__all__ = [
    "CaptWeakPhnConfig",
    "LIMITATIONS",
    "SelectionResult",
    "attention_weights",
    "attn_pool",
    "batch_gop_features",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "figure_3_test_pcc",
    "framework_card",
    "gop_feature_vector",
    "gop_from_log_posteriors",
    "headline_results",
    "mean_pool",
    "pool_over_spans",
    "select_balanced_by_bins",
    "select_unbalanced",
    "simulate_finetune_gain",
    "table_1_dev_results",
]
