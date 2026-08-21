"""DMA-KWS user-defined keyword spotting stub (arXiv:2605.22120)."""

from ltx_trainer.dma_kws.adaptation import apply_lora_update, lora_param_count
from ltx_trainer.dma_kws.config import DmaKwsConfig
from ltx_trainer.dma_kws.ctc_streaming import (
    BLANK_ID,
    blank_insert,
    ctc_streaming_score,
    find_candidate_segments,
)
from ltx_trainer.dma_kws.enrollment import (
    audio_enroll,
    mam_fuse_concat,
    mam_fuse_cross_attention,
    text_enroll,
)
from ltx_trainer.dma_kws.layout import LIMITATIONS
from ltx_trainer.dma_kws.matcher import cosine_similarity, phoneme_match_score, two_stage_score
from ltx_trainer.dma_kws.mock import evaluation_smoke
from ltx_trainer.dma_kws.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_i_si_kws_main,
    table_ii_dual_scaling,
    table_iv_sd_kws,
    table_v_hey_snips_zero_shot,
    table_xii_inference_time,
)

__all__ = [
    "BLANK_ID",
    "LIMITATIONS",
    "DmaKwsConfig",
    "apply_lora_update",
    "audio_enroll",
    "benchmarks_bundle",
    "blank_insert",
    "cosine_similarity",
    "ctc_streaming_score",
    "evaluation_demo",
    "evaluation_smoke",
    "find_candidate_segments",
    "framework_card",
    "headline_results",
    "lora_param_count",
    "mam_fuse_concat",
    "mam_fuse_cross_attention",
    "phoneme_match_score",
    "table_i_si_kws_main",
    "table_ii_dual_scaling",
    "table_iv_sd_kws",
    "table_v_hey_snips_zero_shot",
    "table_xii_inference_time",
    "text_enroll",
    "two_stage_score",
]

