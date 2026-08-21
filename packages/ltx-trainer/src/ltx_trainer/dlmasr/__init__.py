"""DLM-ASR decoding strategies — confidence thresholding (arXiv:2605.29613)."""

from ltx_trainer.dlmasr.config import DecodingStrategy, DlmAsrConfig
from ltx_trainer.dlmasr.decoding import (
    block_decode_smoke,
    decode_round,
    dynamic_threshold_commit,
    fixed_number_commit,
    static_threshold_commit,
    token_confidence,
)
from ltx_trainer.dlmasr.layout import LIMITATIONS
from ltx_trainer.dlmasr.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_i_wer_rtf_tradeoff,
    figure_ii_matched_rtf_wer,
    figure_iii_throughput,
    figure_iv_confidence_ccdf,
    framework_card,
    headline_results,
    pipeline_demo,
)
from ltx_trainer.dlmasr.strategy_compare import compare_strategies, strategy_compare_smoke
from ltx_trainer.dlmasr.uncertainty import (
    ar_reference_uncertainty,
    cumulative_uncertainty,
    token_nll,
    uncertainty_gap_at_progress,
    uncertainty_smoke,
)

__all__ = [
    "DecodingStrategy",
    "DlmAsrConfig",
    "LIMITATIONS",
    "compare_strategies",
    "strategy_compare_smoke",
    "ar_reference_uncertainty",
    "benchmarks_bundle",
    "block_decode_smoke",
    "cumulative_uncertainty",
    "decode_round",
    "dynamic_threshold_commit",
    "evaluation_demo",
    "figure_i_wer_rtf_tradeoff",
    "figure_ii_matched_rtf_wer",
    "figure_iii_throughput",
    "figure_iv_confidence_ccdf",
    "fixed_number_commit",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "static_threshold_commit",
    "token_confidence",
    "token_nll",
    "uncertainty_gap_at_progress",
    "uncertainty_smoke",
]
