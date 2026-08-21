"""CPC3 WLAM text-assisted intelligibility stub."""

from ltx_trainer.cpc3_wlam.alignment import (
    select_dynamic_topk_heads,
    sharpness_score,
    utterance_global_summary,
    word_aligned_local_summary,
)
from ltx_trainer.cpc3_wlam.config import Cpc3WlamConfig
from ltx_trainer.cpc3_wlam.fusion import joint_fuse, predict_word_correctness, sigmoid
from ltx_trainer.cpc3_wlam.layout import LIMITATIONS
from ltx_trainer.cpc3_wlam.mock import evaluation_smoke
from ltx_trainer.cpc3_wlam.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_i_main_comparison,
    table_ii_severity,
    table_iii_diagnostics,
)
from ltx_trainer.cpc3_wlam.word_level import (
    aggregate_word_states,
    masked_bce_loss,
    sentence_intelligibility_from_word_probs,
)

__all__ = [
    "Cpc3WlamConfig",
    "LIMITATIONS",
    "aggregate_word_states",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "joint_fuse",
    "masked_bce_loss",
    "predict_word_correctness",
    "select_dynamic_topk_heads",
    "sentence_intelligibility_from_word_probs",
    "sharpness_score",
    "sigmoid",
    "table_i_main_comparison",
    "table_ii_severity",
    "table_iii_diagnostics",
    "utterance_global_summary",
    "word_aligned_local_summary",
]
