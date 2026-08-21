"""Before the Shutter — 3D aesthetic portrait planning (arXiv:2605.30318).

Reference math for Photographic Scene Graph, comparative planning frontier,
thin-lens exposure, and paper Table 1–2 metrics. Full Blender + MLLM stack is external.
"""

from ltx_trainer.before_shutter.camera import (
    ev100_from_aperture_shutter,
    exposure_validity_logit,
    meter_shutter_from_ev100,
    plan_camera_from_graph_constraint,
    render_linear_radiance,
)
from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState
from ltx_trainer.before_shutter.human import (
    balance_score,
    batch_actionability,
    collision_penetration,
    realize_staging,
)
from ltx_trainer.before_shutter.lighting import (
    LightDevice,
    apply_preset,
    devices_from_state,
    lighting_ratio,
    refine_lighting_step,
)
from ltx_trainer.before_shutter.metrics import (
    bradley_terry_win_probability,
    evaluate_plan_batch,
    fit_bradley_terry_from_wins,
    table1_main_results,
    table2_stage_ablation,
)
from ltx_trainer.before_shutter.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    table_baselines,
    training_step_demo,
)
from ltx_trainer.before_shutter.planning import (
    AestheticFrontier,
    FrontierEntry,
    judge_pairwise_preference,
    mock_viewfinder_score,
    planning_step_composition,
    planning_step_lighting,
    planning_step_staging,
    run_comparative_planning,
)
from ltx_trainer.before_shutter.scene_graph import (
    EmissiveEdge,
    PhotographicSceneGraph,
    SceneNode,
    SpatialEdge,
    build_demo_graph,
    probe_emitter_ratios,
)

__all__ = [
    "AestheticFrontier",
    "BeforeShutterConfig",
    "EmissiveEdge",
    "FrontierEntry",
    "LightDevice",
    "PhotographicSceneGraph",
    "PortraitPlanState",
    "SceneNode",
    "SpatialEdge",
    "apply_preset",
    "balance_score",
    "batch_actionability",
    "benchmark_manifest",
    "bradley_terry_win_probability",
    "build_demo_graph",
    "collision_penetration",
    "devices_from_state",
    "evaluate_plan_batch",
    "evaluation_demo",
    "ev100_from_aperture_shutter",
    "exposure_validity_logit",
    "fit_bradley_terry_from_wins",
    "framework_card",
    "judge_pairwise_preference",
    "lighting_ratio",
    "meter_shutter_from_ev100",
    "mock_viewfinder_score",
    "paper_limitations",
    "plan_camera_from_graph_constraint",
    "planning_step_composition",
    "planning_step_lighting",
    "planning_step_staging",
    "probe_emitter_ratios",
    "realize_staging",
    "refine_lighting_step",
    "render_linear_radiance",
    "run_comparative_planning",
    "table1_main_results",
    "table2_stage_ablation",
    "table_baselines",
    "training_step_demo",
]
