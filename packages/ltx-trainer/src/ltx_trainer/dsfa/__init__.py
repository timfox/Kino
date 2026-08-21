"""DSFA — proxy-to-wild CodecFake detection (arXiv:2606.07494)."""

from ltx_trainer.dsfa.augmentation import dsfa_demo, dsfa_transform, maybe_apply_dsfa
from ltx_trainer.dsfa.config import DsfaConfig
from ltx_trainer.dsfa.domain_gap import proxy_wild_gap_demo
from ltx_trainer.dsfa.eval import eval_smoke, pipeline_demo
from ltx_trainer.dsfa.fold import annotate_audio_save_data
from ltx_trainer.dsfa.losses import loss_demo, total_loss
from ltx_trainer.dsfa.mock import evaluation_smoke
from ltx_trainer.dsfa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_cosg_datasets,
    table2_main_results,
    table3_layer_ablation,
    table4_dsfa_probability,
)

__all__ = [
    "DsfaConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "dsfa_demo",
    "dsfa_transform",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "loss_demo",
    "maybe_apply_dsfa",
    "pipeline_demo",
    "proxy_wild_gap_demo",
    "table1_cosg_datasets",
    "table2_main_results",
    "table3_layer_ablation",
    "table4_dsfa_probability",
    "total_loss",
]
