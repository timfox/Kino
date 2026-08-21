"""Paris 2.0 decentralized video DDM (Rouzbayani et al., arXiv:2605.26064)."""

from ltx_trainer.paris2.config import Paris2Config
from ltx_trainer.paris2.experts import ExpertPool, alternating_schedule_weights, top_k_mask
from ltx_trainer.paris2.flow_matching import (
    ExpertVelocityStub,
    flow_matching_loss,
    sample_linear_path,
)
from ltx_trainer.paris2.router import Paris2Router, pool_latent
from ltx_trainer.paris2.pipeline import (
    evaluation_demo,
    framework_card,
    table_expert_specialization,
    table_relative_improvement,
    table_stage1_t2v,
    table_switching_schedule_ablation,
    training_step_demo,
)

__all__ = [
    "Paris2Config",
    "Paris2Router",
    "ExpertPool",
    "ExpertVelocityStub",
    "alternating_schedule_weights",
    "evaluation_demo",
    "flow_matching_loss",
    "framework_card",
    "pool_latent",
    "sample_linear_path",
    "table_expert_specialization",
    "table_relative_improvement",
    "table_stage1_t2v",
    "table_switching_schedule_ablation",
    "top_k_mask",
    "training_step_demo",
]
