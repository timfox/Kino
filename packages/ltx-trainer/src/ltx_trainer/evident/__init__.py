"""EVIDENT entity-grounded cross-domain VTG (Ahn et al., arXiv:2605.26104)."""

from ltx_trainer.evident.config import EVIDENTConfig
from ltx_trainer.evident.e2v_gating import e2v_gating_scores, minmax_normalize
from ltx_trainer.evident.eb_adapter import EntityBottleneckAdapter, SlotAttentionBlock
from ltx_trainer.evident.eb_distillation import (
    entity_binding_distillation_loss,
    kmeans_cluster_maps,
)
from ltx_trainer.evident.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_adapter_design,
    table_ablation_components,
    table_cross_domain_vbench,
    table_domain_gap_analysis,
    training_step_demo,
)

__all__ = [
    "EVIDENTConfig",
    "EntityBottleneckAdapter",
    "SlotAttentionBlock",
    "e2v_gating_scores",
    "entity_binding_distillation_loss",
    "evaluation_demo",
    "framework_card",
    "kmeans_cluster_maps",
    "minmax_normalize",
    "table_ablation_adapter_design",
    "table_ablation_components",
    "table_cross_domain_vbench",
    "table_domain_gap_analysis",
    "training_step_demo",
]
