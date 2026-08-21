"""R-FUML: Robust Fuzzy Multi-View Learning under view conflict (Duan et al., arXiv:2605.24475)."""

from ltx_trainer.rfuml.config import RFUMLConfig
from ltx_trainer.rfuml.conflict import conflict_for_view
from ltx_trainer.rfuml.fuzzy import (
    category_credibility,
    gamma_schedule,
    logits_to_memberships,
    necessity_from_membership,
    training_credibility,
    uncertainty_from_credibility,
)
from ltx_trainer.rfuml.fusion import fuse_memberships, rmf_weights
from ltx_trainer.rfuml.layout import LIMITATIONS
from ltx_trainer.rfuml.loss import lccl, lrccl, total_loss
from ltx_trainer.rfuml.mock import MultiViewInstance, make_clean_instance, make_conflicting_instance
from ltx_trainer.rfuml.pipeline import (
    evaluation_demo,
    framework_card,
    rmf_demo,
    rlvc_demo,
    table_ablation_loss_fusion,
    table_ablation_rlvc_stages,
    table_accuracy_excerpt,
    table_degradation_hw,
    table_fpr95,
    training_step_demo,
)
from ltx_trainer.rfuml.rlvc import (
    GMMPartition,
    cyclical_learning_rate,
    fit_gmm_two_component,
    importance_weight,
    per_view_average_loss,
)

__all__ = [
    "GMMPartition",
    "LIMITATIONS",
    "MultiViewInstance",
    "RFUMLConfig",
    "category_credibility",
    "conflict_for_view",
    "cyclical_learning_rate",
    "evaluation_demo",
    "fit_gmm_two_component",
    "framework_card",
    "fuse_memberships",
    "gamma_schedule",
    "importance_weight",
    "lccl",
    "logits_to_memberships",
    "lrccl",
    "make_clean_instance",
    "make_conflicting_instance",
    "necessity_from_membership",
    "per_view_average_loss",
    "rmf_demo",
    "rmf_weights",
    "rlvc_demo",
    "table_ablation_loss_fusion",
    "table_ablation_rlvc_stages",
    "table_accuracy_excerpt",
    "table_degradation_hw",
    "table_fpr95",
    "total_loss",
    "training_credibility",
    "training_step_demo",
    "uncertainty_from_credibility",
]
