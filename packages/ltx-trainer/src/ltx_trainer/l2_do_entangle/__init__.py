"""Dual-output L2 ASR MTL entanglement (arXiv:2606.06065)."""

from ltx_trainer.l2_do_entangle.analysis import linear_cka, stratified_cer_gaps
from ltx_trainer.l2_do_entangle.config import L2DoEntangleConfig
from ltx_trainer.l2_do_entangle.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.l2_do_entangle.fold import annotate_audio_save_data
from ltx_trainer.l2_do_entangle.losses import cer_gap, dual_output_loss, single_output_loss
from ltx_trainer.l2_do_entangle.mock import evaluation_smoke
from ltx_trainer.l2_do_entangle.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure2_stratified_gaps,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_dataset_stats,
    table2_cer_results,
    table3_encoder_cka,
    table4_decoder_cka,
)

__all__ = [
    "L2DoEntangleConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "cer_gap",
    "dual_output_loss",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "figure2_stratified_gaps",
    "framework_card",
    "headline_results",
    "linear_cka",
    "pipeline_demo",
    "pipeline_demo_export",
    "single_output_loss",
    "stratified_cer_gaps",
    "table1_dataset_stats",
    "table2_cer_results",
    "table3_encoder_cka",
    "table4_decoder_cka",
]
