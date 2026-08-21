"""Pantheon360: 3D-aware 360° video diffusion (Chen et al., arXiv:2605.25449)."""

from ltx_trainer.pantheon360.cache import PointCloudCache, build_cache_from_erp, render_trajectory_erp
from ltx_trainer.pantheon360.conditioning import (
    clip_perspective_crops,
    concat_geometric_latent,
    diffusion_loss_stub,
    encode_geometry_scaffold,
    semantic_condition_vector,
)
from ltx_trainer.pantheon360.config import Pantheon360Config
from ltx_trainer.pantheon360.fusion import dual_anchor_latent_fusion
from ltx_trainer.pantheon360.integration import proceduralsky_card, proceduralsky_pipeline_plan
from ltx_trainer.pantheon360.ltx_plan import gopex_env_exports, ltx_training_plan
from ltx_trainer.pantheon360.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    paper_checks,
    table_cache_ablation_habitat,
    table_cache_ablation_web360,
    table_habitat_sparse_views,
    table_latent_fusion_ablation,
    table_runtime_a100,
    table_web360_single_view,
    training_step_demo,
)

__all__ = [
    "Pantheon360Config",
    "PointCloudCache",
    "build_cache_from_erp",
    "clip_perspective_crops",
    "concat_geometric_latent",
    "diffusion_loss_stub",
    "dual_anchor_latent_fusion",
    "encode_geometry_scaffold",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "gopex_env_exports",
    "ltx_training_plan",
    "paper_checks",
    "proceduralsky_card",
    "proceduralsky_pipeline_plan",
    "render_trajectory_erp",
    "semantic_condition_vector",
    "table_cache_ablation_habitat",
    "table_cache_ablation_web360",
    "table_habitat_sparse_views",
    "table_latent_fusion_ablation",
    "table_runtime_a100",
    "table_web360_single_view",
    "training_step_demo",
]
