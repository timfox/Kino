"""FIGMA — fine-grained music retrieval (arXiv:2606.06615)."""

from ltx_trainer.figma.caption import caption_saturation_demo, saturating_recall_curve
from ltx_trainer.figma.config import FigmaConfig
from ltx_trainer.figma.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.figma.fold import annotate_audio_save_data
from ltx_trainer.figma.loss import (
    frame_contrastive_loss,
    frame_level_score,
    global_contrastive_loss,
    loss_demo,
    multi_view_loss,
)
from ltx_trainer.figma.mock import evaluation_smoke
from ltx_trainer.figma.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    experimental_protocol,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_dataset_comparison,
    table2_musicbench,
    table3_fmacaps_eval,
    table4_perturbation_a2t,
    table5_fgmcaps_test,
    table8_source_splits,
)

__all__ = [
    "FigmaConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "caption_saturation_demo",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "experimental_protocol",
    "frame_contrastive_loss",
    "frame_level_score",
    "framework_card",
    "global_contrastive_loss",
    "headline_results",
    "loss_demo",
    "multi_view_loss",
    "pipeline_demo",
    "pipeline_demo_export",
    "saturating_recall_curve",
    "table1_dataset_comparison",
    "table2_musicbench",
    "table3_fmacaps_eval",
    "table4_perturbation_a2t",
    "table5_fgmcaps_test",
    "table8_source_splits",
]
