"""Thaka KSAA-2026 Arabic speech diacritization (CATT-Whisper + regularization)."""

from ltx_trainer.ksaa_diac.architecture import (
    architecture_summary,
    mean_pool_whisper_frames,
    prefix_add_fusion,
)
from ltx_trainer.ksaa_diac.config import KsaaDiacConfig
from ltx_trainer.ksaa_diac.inference import (
    ensemble_softmax_average,
    mc_dropout_average,
    predict_diacritics,
)
from ltx_trainer.ksaa_diac.layout import LIMITATIONS
from ltx_trainer.ksaa_diac.losses import focal_loss, rdrop_loss, symmetric_kl
from ltx_trainer.ksaa_diac.mock import evaluation_smoke
from ltx_trainer.ksaa_diac.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_hyperparameters,
    table_ii_leaderboard,
    table_iii_ablation,
    table_iv_example_snippet,
)
from ltx_trainer.ksaa_diac.postprocess import insert_diacritics, strip_diacritics, verify_invariants

__all__ = [
    "LIMITATIONS",
    "KsaaDiacConfig",
    "architecture_summary",
    "benchmarks_bundle",
    "ensemble_softmax_average",
    "evaluation_demo",
    "evaluation_smoke",
    "focal_loss",
    "framework_card",
    "headline_results",
    "insert_diacritics",
    "mc_dropout_average",
    "mean_pool_whisper_frames",
    "pipeline_demo",
    "predict_diacritics",
    "prefix_add_fusion",
    "rdrop_loss",
    "strip_diacritics",
    "symmetric_kl",
    "table_i_hyperparameters",
    "table_ii_leaderboard",
    "table_iii_ablation",
    "table_iv_example_snippet",
    "verify_invariants",
]
