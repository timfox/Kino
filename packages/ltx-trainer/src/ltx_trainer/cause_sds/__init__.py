"""Cause-aware error diagnosis for cascaded ASR-LLM spoken dialogue systems."""

from ltx_trainer.cause_sds.clarification import (
    clarification_query,
    k_round_clarification,
    run_clarification_round,
    strategy_for_cause,
)
from ltx_trainer.cause_sds.config import CauseSdsConfig
from ltx_trainer.cause_sds.detectors import (
    DistortionEvent,
    ErrorCause,
    deletion_mask_frame,
    detector_profile,
    fuse_error_causes,
    predict_binary,
    tag_span,
)
from ltx_trainer.cause_sds.layout import LIMITATIONS
from ltx_trainer.cause_sds.metrics import recall_fpr, wer, werr
from ltx_trainer.cause_sds.mock import evaluation_smoke
from ltx_trainer.cause_sds.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_token_detection,
    table_ii_word_detection,
    table_iii_distortion_events,
    table_iv_wer_clarification,
    table_v_dialogue_maj,
)
from ltx_trainer.cause_sds.tsallis import (
    is_error_word,
    tsallis_token_confidence,
    word_confidence,
)

__all__ = [
    "LIMITATIONS",
    "CauseSdsConfig",
    "DistortionEvent",
    "ErrorCause",
    "benchmarks_bundle",
    "clarification_query",
    "deletion_mask_frame",
    "detector_profile",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "fuse_error_causes",
    "headline_results",
    "is_error_word",
    "k_round_clarification",
    "pipeline_demo",
    "predict_binary",
    "recall_fpr",
    "run_clarification_round",
    "strategy_for_cause",
    "table_i_token_detection",
    "table_ii_word_detection",
    "table_iii_distortion_events",
    "table_iv_wer_clarification",
    "table_v_dialogue_maj",
    "tag_span",
    "tsallis_token_confidence",
    "wer",
    "werr",
    "word_confidence",
]
