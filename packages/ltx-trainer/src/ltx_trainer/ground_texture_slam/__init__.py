"""Multi-session ground texture SLAM (Hart & Englot arXiv:2605.19701)."""

from ltx_trainer.ground_texture_slam.kld import channel_histogram, kld_rgb, scale_covariance_kld
from ltx_trainer.ground_texture_slam.metrics import position_rmse, table_i_rmse
from ltx_trainer.ground_texture_slam.pipeline import GroundTextureSLAM, SLAMConfig, evaluate_sessions
from ltx_trainer.ground_texture_slam.schema import LoopClosureMethod, Pose2D

__all__ = [
    "GroundTextureSLAM",
    "LoopClosureMethod",
    "Pose2D",
    "SLAMConfig",
    "channel_histogram",
    "evaluate_sessions",
    "kld_rgb",
    "position_rmse",
    "scale_covariance_kld",
    "table_i_rmse",
]
