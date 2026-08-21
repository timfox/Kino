"""Swarical: hierarchical localization for Flying Light Specks (Alimohammadzadeh & Ghandeharizadeh, MM '24)."""

from ltx_trainer.swarical.config import (
    FlsCameraOrientation,
    LocalizationMode,
    RaspberryCameraSpec,
    SwaricalConfig,
    camera_performance_table2,
    camera_specs_table1,
    fls_density_per_area,
    swarm_count,
)
from ltx_trainer.swarical.layout import LIMITATIONS
from ltx_trainer.swarical.localization import (
    PoseVector,
    apply_translation,
    average_correction,
    center_align,
    correction_vector,
    intra_swarm_step,
    is_converged,
    localization_mode_summary,
)
from ltx_trainer.swarical.metrics import (
    chamfer_distance,
    estimate_hd_from_camera_error,
    hausdorff_distance,
    shrink_point_cloud,
)
from ltx_trainer.swarical.planner import (
    FlSTreeEdge,
    PlannerOutput,
    Point3,
    SwarmTreeEdge,
    build_fls_tree,
    build_swarm_tree,
    estimate_fls_for_face_area,
    plan_from_points,
    swarm_centers,
)
from ltx_trainer.swarical.aruco_pose import (
    aruco_reproduction_guide,
    classify_detection_range,
    pi_pose_estimation_command,
    sample_distance_error_table,
)
from ltx_trainer.swarical.datasets import get_mesh_dataset, list_mesh_datasets
from ltx_trainer.swarical.dead_reckoning import apply_dead_reckoning
from ltx_trainer.swarical.online_sim import compare_localization_modes, run_small_scale_experiment
from ltx_trainer.swarical.pipeline import (
    evaluation_demo,
    framework_card,
    localization_demo,
    metrics_demo,
    planner_demo,
    table_camera_performance,
    table_camera_specs,
    table_group_size_sensitivity,
    table_localization_skateboard_g50,
    table_skateboard_planner,
    table_swarmer_comparison,
)
from ltx_trainer.swarical.reproducibility import (
    artifact_manifest,
    companion_paper_metadata,
    reproduction_walkthrough,
)

__all__ = [
    "FlSTreeEdge",
    "FlsCameraOrientation",
    "LIMITATIONS",
    "LocalizationMode",
    "PlannerOutput",
    "Point3",
    "PoseVector",
    "RaspberryCameraSpec",
    "SwarmTreeEdge",
    "SwaricalConfig",
    "apply_translation",
    "average_correction",
    "build_fls_tree",
    "build_swarm_tree",
    "camera_performance_table2",
    "camera_specs_table1",
    "center_align",
    "chamfer_distance",
    "correction_vector",
    "estimate_fls_for_face_area",
    "estimate_hd_from_camera_error",
    "evaluation_demo",
    "fls_density_per_area",
    "framework_card",
    "get_mesh_dataset",
    "hausdorff_distance",
    "intra_swarm_step",
    "is_converged",
    "localization_demo",
    "list_mesh_datasets",
    "localization_mode_summary",
    "metrics_demo",
    "pi_pose_estimation_command",
    "reproduction_walkthrough",
    "run_small_scale_experiment",
    "sample_distance_error_table",
    "plan_from_points",
    "planner_demo",
    "shrink_point_cloud",
    "swarm_centers",
    "swarm_count",
    "table_camera_performance",
    "table_camera_specs",
    "table_group_size_sensitivity",
    "table_localization_skateboard_g50",
    "table_skateboard_planner",
    "table_swarmer_comparison",
]
