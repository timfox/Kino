"""PINN synchronization control stub (arXiv:2601.00178)."""

from ltx_trainer.sync_pinn.config import SyncPinnConfig
from ltx_trainer.sync_pinn.kuramoto import (
    euler_rollout,
    frequency_compensation,
    full_adjacency,
    kuramoto_rhs,
    order_parameter,
    phase_feedback,
    sakaguchi_rhs,
    shaping_h,
)
from ltx_trainer.sync_pinn.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.sync_pinn.losses import (
    composite_training_loss,
    dynamics_residual_norm,
    initial_condition_loss,
    persistence_control_loss,
    regulation_loss,
)
from ltx_trainer.sync_pinn.metrics import (
    instantaneous_control_cost,
    integrated_control_cost,
    persistence_satisfied,
    relative_sync_error,
    time_averaged_order,
)
from ltx_trainer.sync_pinn.mock import evaluation_smoke
from ltx_trainer.sync_pinn.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.sync_pinn.tables import headline_results

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "SyncPinnConfig",
    "benchmarks_bundle",
    "composite_training_loss",
    "dynamics_residual_norm",
    "euler_rollout",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "frequency_compensation",
    "full_adjacency",
    "headline_results",
    "initial_condition_loss",
    "instantaneous_control_cost",
    "integrated_control_cost",
    "kuramoto_rhs",
    "order_parameter",
    "persistence_control_loss",
    "persistence_satisfied",
    "phase_feedback",
    "regulation_loss",
    "relative_sync_error",
    "sakaguchi_rhs",
    "shaping_h",
    "time_averaged_order",
]
