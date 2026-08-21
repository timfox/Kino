"""Group-action world models: GA regularization + GAC/GAR metrics (Wang et al., arXiv:2605.24578)."""

from ltx_trainer.group_action.config import GroupActionConfig
from ltx_trainer.group_action.losses import (
    composition_loss,
    group_action_loss,
    identity_loss,
    inverse_loss,
)
from ltx_trainer.group_action.metrics import (
    composition_probe_error,
    gac_aggregate,
    gar_error,
    identity_probe_error,
    inverse_probe_error,
)
from ltx_trainer.group_action.pipeline import (
    evaluate_gac_from_probes,
    evaluate_gar_from_rollouts,
    ga_training_step,
)
from ltx_trainer.group_action.se2 import (
    accumulate_actions,
    composition_alternative_segment,
    inverse_segment,
    negate_action,
    zero_action,
)
from ltx_trainer.group_action.state import state_distance, state_from_xytheta
from ltx_trainer.group_action.synthesis import identity_segment, synthesize_constraint

__all__ = [
    "GroupActionConfig",
    "accumulate_actions",
    "composition_alternative_segment",
    "composition_loss",
    "composition_probe_error",
    "evaluate_gac_from_probes",
    "evaluate_gar_from_rollouts",
    "gac_aggregate",
    "ga_training_step",
    "gar_error",
    "group_action_loss",
    "identity_loss",
    "identity_probe_error",
    "identity_segment",
    "inverse_probe_error",
    "inverse_loss",
    "inverse_segment",
    "negate_action",
    "state_distance",
    "state_from_xytheta",
    "synthesize_constraint",
    "zero_action",
]
