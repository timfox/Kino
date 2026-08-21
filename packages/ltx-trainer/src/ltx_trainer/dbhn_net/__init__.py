"""DBHN-Net dual-branch hybrid speech enhancement (arXiv:2606.05911)."""

from ltx_trainer.dbhn_net.config import DbhnNetConfig
from ltx_trainer.dbhn_net.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.dbhn_net.lif import gradient_proxy, lif_step
from ltx_trainer.dbhn_net.loss import combined_loss
from ltx_trainer.dbhn_net.mock import evaluation_smoke
from ltx_trainer.dbhn_net.modules import (
    band_merge_bands,
    band_split_spectrum,
    information_transformation_block,
    interaction_block,
    tf_cross_attention_fusion,
)
from ltx_trainer.dbhn_net.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table2_dual_branch,
    table3_mamba_ablation,
    table6_wsj0,
    table7_voicebank,
    table8_dns,
    table9_complexity,
)

__all__ = [
    "DbhnNetConfig",
    "annotate_audio_save_data",
    "band_merge_bands",
    "band_split_spectrum",
    "benchmarks_bundle",
    "combined_loss",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "gradient_proxy",
    "headline_results",
    "information_transformation_block",
    "interaction_block",
    "lif_step",
    "pipeline_demo",
    "pipeline_demo_export",
    "table2_dual_branch",
    "table3_mamba_ablation",
    "table6_wsj0",
    "table7_voicebank",
    "table8_dns",
    "table9_complexity",
    "tf_cross_attention_fusion",
]

from ltx_trainer.dbhn_net.fold import annotate_audio_save_data  # noqa: E402
