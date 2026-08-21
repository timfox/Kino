"""VTM partition acceleration survey + RL stub (arXiv:2605.21526)."""

from ltx_trainer.vtm_partition.config import VTMPartitionConfig
from ltx_trainer.vtm_partition.features import CUContext, extract_state_vector, feature_groups
from ltx_trainer.vtm_partition.layout import LIMITATIONS
from ltx_trainer.vtm_partition.metrics import (
    complexity_reduction_pct,
    complexity_ratio,
    encoding_time_ratio,
    relative_complexity_pct,
)
from ltx_trainer.vtm_partition.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    ra_generalization_excerpt,
    sota_partition_methods,
    table_i_vtm_evolution,
    table_ii_rl_features,
)
from ltx_trainer.vtm_partition.qtmtt import ALL_SPLITS, hevc_vs_vvc_scale_facts
from ltx_trainer.vtm_partition.rl_agent import (
    composite_loss,
    predict_q_values_linear,
    rl_tradeoff_point,
    select_splits_top_n,
    table_iii_rl_tradeoffs,
)

__all__ = [
    "ALL_SPLITS",
    "CUContext",
    "LIMITATIONS",
    "VTMPartitionConfig",
    "benchmarks_bundle",
    "complexity_ratio",
    "complexity_reduction_pct",
    "composite_loss",
    "encoding_time_ratio",
    "relative_complexity_pct",
    "evaluation_demo",
    "extract_state_vector",
    "feature_groups",
    "framework_card",
    "hevc_vs_vvc_scale_facts",
    "pipeline_demo",
    "predict_q_values_linear",
    "ra_generalization_excerpt",
    "rl_tradeoff_point",
    "select_splits_top_n",
    "sota_partition_methods",
    "table_i_vtm_evolution",
    "table_ii_rl_features",
    "table_iii_rl_tradeoffs",
]
