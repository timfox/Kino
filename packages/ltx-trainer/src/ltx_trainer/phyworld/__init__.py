"""PhyWorld — physics-faithful video world model (arXiv:2605.19242)."""

from ltx_trainer.phyworld.config import PhyWorldConfig
from ltx_trainer.phyworld.dpo import (
    aggregate_human_score,
    diffusion_dpo_loss,
    preference_logit,
    preference_margin_ok,
    sample_timestep_high_noise,
)
from ltx_trainer.phyworld.flow_matching import (
    flow_matching_loss,
    interpolate_latent,
    target_velocity,
)
from ltx_trainer.phyworld.judge import (
    PHYSICS_EVENT_CLASSES,
    judge_dimensions,
    overall_physics_score,
    physics_filter_decision,
    round4_trainset_quotas,
)
from ltx_trainer.phyworld.layout import LIMITATIONS
from ltx_trainer.phyworld.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    preference_pipeline_summary,
    table_iii_vbench,
    table_iv_physics_faithfulness,
)
from ltx_trainer.phyworld.v2v import build_frame_mask, concat_conditioning_channels

__all__ = [
    "LIMITATIONS",
    "PHYSICS_EVENT_CLASSES",
    "PhyWorldConfig",
    "aggregate_human_score",
    "benchmarks_bundle",
    "build_frame_mask",
    "concat_conditioning_channels",
    "diffusion_dpo_loss",
    "evaluation_demo",
    "flow_matching_loss",
    "framework_card",
    "interpolate_latent",
    "judge_dimensions",
    "overall_physics_score",
    "physics_filter_decision",
    "pipeline_demo",
    "preference_logit",
    "preference_margin_ok",
    "preference_pipeline_summary",
    "round4_trainset_quotas",
    "sample_timestep_high_noise",
    "table_iii_vbench",
    "table_iv_physics_faithfulness",
    "target_velocity",
]
