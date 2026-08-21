"""AVBench: human-aligned automated evaluation for audio-video generation (Yang et al., arXiv:2605.24652)."""

from ltx_trainer.avbench.config import AVBenchConfig
from ltx_trainer.avbench.layout import LIMITATIONS
from ltx_trainer.avbench.mock import softmax_pair, toy_logits_aligned, toy_logits_misaligned
from ltx_trainer.avbench.pipeline import (
    evaluation_demo,
    framework_card,
    table_benchmark_comparison,
    table_evaluator_hard_negative_accuracy,
    table_human_2afc_accuracy,
    table_human_alignment_pearson,
    table_t2av_main_results,
    table_vt_consistency_instance_accuracy,
    training_step_demo,
)
from ltx_trainer.avbench.scoring import (
    audiobox_aesthetic_score,
    model_win_ratio,
    pearson_r,
    speech_content_score,
    yes_no_alignment_score,
)
from ltx_trainer.avbench.taxonomy import (
    AT_PRIMARY_CATEGORIES,
    AV_NEGATIVE_STRATEGIES,
    SUITE_DIMENSIONS,
    VT_DIMENSIONS,
)

__all__ = [
    "AT_PRIMARY_CATEGORIES",
    "AVBenchConfig",
    "AV_NEGATIVE_STRATEGIES",
    "LIMITATIONS",
    "SUITE_DIMENSIONS",
    "VT_DIMENSIONS",
    "audiobox_aesthetic_score",
    "evaluation_demo",
    "framework_card",
    "model_win_ratio",
    "pearson_r",
    "softmax_pair",
    "speech_content_score",
    "table_benchmark_comparison",
    "table_evaluator_hard_negative_accuracy",
    "table_human_2afc_accuracy",
    "table_human_alignment_pearson",
    "table_t2av_main_results",
    "table_vt_consistency_instance_accuracy",
    "toy_logits_aligned",
    "toy_logits_misaligned",
    "training_step_demo",
    "yes_no_alignment_score",
]
