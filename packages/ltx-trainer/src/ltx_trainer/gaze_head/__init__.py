"""Gaze-head coordination: cVAE head motion from gaze (Liu et al., arXiv:2605.25810).

Reference: pitch/yaw poses, Eq. (1) cVAE loss, GRU temporal model, CelebV-Text metrics (Table 1–2).
Full in-the-wild extraction (MediaPipe, PySceneDetect, Qin gaze) and ST-ED video synthesis are external.
"""

from ltx_trainer.gaze_head.baselines import constant_head_motion, mirror_gaze_inputs
from ltx_trainer.gaze_head.config import GazeHeadConfig
from ltx_trainer.gaze_head.losses import cvae_loss, kl_divergence_gaussian
from ltx_trainer.gaze_head.metrics import (
    average_pairwise_distance,
    average_variance_error,
    correlation_pitch_yaw,
    evaluate_sequence,
    pearson_correlation,
    smoothness_jerk,
)
from ltx_trainer.gaze_head.model import GazeHeadCoordinationCVAE
from ltx_trainer.gaze_head.pipeline import (
    data_pipeline_stages,
    evaluation_demo,
    framework_card,
    table_human_evaluation,
    table_quantitative_comparison,
    training_step_demo,
)
from ltx_trainer.gaze_head.pose import angular_error_deg, directions_from_pose, pitch_yaw_to_direction

__all__ = [
    "GazeHeadConfig",
    "GazeHeadCoordinationCVAE",
    "angular_error_deg",
    "average_pairwise_distance",
    "average_variance_error",
    "constant_head_motion",
    "correlation_pitch_yaw",
    "cvae_loss",
    "data_pipeline_stages",
    "directions_from_pose",
    "evaluate_sequence",
    "evaluation_demo",
    "framework_card",
    "kl_divergence_gaussian",
    "mirror_gaze_inputs",
    "pearson_correlation",
    "pitch_yaw_to_direction",
    "smoothness_jerk",
    "table_human_evaluation",
    "table_quantitative_comparison",
    "training_step_demo",
]
