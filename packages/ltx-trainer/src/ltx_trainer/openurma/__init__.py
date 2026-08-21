"""OpenURMA — clean-room Unified Bus implementation stub (arXiv:2605.28717)."""

from ltx_trainer.openurma.config import OpenURMAConfig
from ltx_trainer.openurma.latency import headline_ratios, latency_decomposition, stacks_summary
from ltx_trainer.openurma.mock import evaluation_smoke
from ltx_trainer.openurma.ordering import ordering_surface_card
from ltx_trainer.openurma.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.openurma.state_model import per_nic_state_ub, state_at_scale, state_scaling_table

__all__ = [
    "OpenURMAConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_ratios",
    "knowledge_card",
    "latency_decomposition",
    "ordering_surface_card",
    "per_nic_state_ub",
    "stacks_summary",
    "state_at_scale",
    "state_scaling_table",
]
