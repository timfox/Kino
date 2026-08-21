"""Trajectory-guided diffusion I2V for maritime frame reconstruction (Bompai et al. arXiv:2605.16420)."""

from ltx_trainer.traj_i2v.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.traj_i2v.config import TrajI2VConfig
from ltx_trainer.traj_i2v.gps_mapping import GpsFix, VesselAnchor, project_gps_to_pixel
from ltx_trainer.traj_i2v.paper import evaluation_demo, framework_card
from ltx_trainer.traj_i2v.pipeline import reconstruct_clip, save_conditioning_json

__all__ = [
    "GpsFix",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "TrajI2VConfig",
    "VesselAnchor",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "project_gps_to_pixel",
    "reconstruct_clip",
    "save_conditioning_json",
]
