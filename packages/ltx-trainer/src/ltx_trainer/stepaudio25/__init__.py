"""StepAudio 2.5 unified audio-language foundation stub."""

from ltx_trainer.stepaudio25.config import Stepaudio25Config
from ltx_trainer.stepaudio25.layout import LIMITATIONS
from ltx_trainer.stepaudio25.mock import evaluation_smoke
from ltx_trainer.stepaudio25.mtp import mtp_branch_weights, mtp_combined_loss, verify_mtp_prefix
from ltx_trainer.stepaudio25.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_4_tts_arena,
    figure_5_realtime_eval,
    framework_card,
    headline_results,
    table_1_asr_error_rates,
    table_2_asr_rtf,
    table_3_mtp_acceptance,
)
from ltx_trainer.stepaudio25.tts_reward import pairwise_grm_score, reward_shaping

__all__ = [
    "LIMITATIONS",
    "Stepaudio25Config",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "figure_4_tts_arena",
    "figure_5_realtime_eval",
    "framework_card",
    "headline_results",
    "mtp_branch_weights",
    "mtp_combined_loss",
    "pairwise_grm_score",
    "reward_shaping",
    "table_1_asr_error_rates",
    "table_2_asr_rtf",
    "table_3_mtp_acceptance",
    "verify_mtp_prefix",
]
