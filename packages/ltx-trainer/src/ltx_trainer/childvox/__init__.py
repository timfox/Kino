"""ChildVox — child-centered audio/speech benchmark (arXiv:2605.29257)."""

from ltx_trainer.childvox.config import ChildVoxCategory, ChildVoxConfig
from ltx_trainer.childvox.datasets import balanced_distribution, dataset_registry
from ltx_trainer.childvox.encoder_proxy import encoder_proxy_smoke, layer_hiddens_from_waveform
from ltx_trainer.childvox.eval_suite import eval_suite_smoke, run_balanced_eval
from ltx_trainer.childvox.layout import LIMITATIONS
from ltx_trainer.childvox.models import lora_smoke, model_catalog, weighted_encoder_pool
from ltx_trainer.childvox.pipeline import (
    application_language_levels,
    application_rhotic_age_correlation,
    benchmarks_bundle,
    evaluation_demo,
    figure_iv_proprietary_comparison,
    framework_card,
    headline_results,
    pipeline_demo,
    table_iii_encoder_macro_f1,
    table_iv_diarization_asr,
    table_v_balanced_subset,
)

__all__ = [
    "ChildVoxCategory",
    "ChildVoxConfig",
    "LIMITATIONS",
    "application_language_levels",
    "application_rhotic_age_correlation",
    "balanced_distribution",
    "benchmarks_bundle",
    "dataset_registry",
    "encoder_proxy_smoke",
    "eval_suite_smoke",
    "evaluation_demo",
    "figure_iv_proprietary_comparison",
    "framework_card",
    "headline_results",
    "layer_hiddens_from_waveform",
    "lora_smoke",
    "model_catalog",
    "pipeline_demo",
    "run_balanced_eval",
    "table_iii_encoder_macro_f1",
    "table_iv_diarization_asr",
    "table_v_balanced_subset",
    "weighted_encoder_pool",
]
