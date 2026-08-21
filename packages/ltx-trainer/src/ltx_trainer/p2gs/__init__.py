"""P2GS: Physical Prior-guided Gaussian Splatting (Shimomura et al. arXiv:2605.16925)."""

from ltx_trainer.p2gs.cameras import Camera, load_scene_cameras
from ltx_trainer.p2gs.gaussians import GaussianModel
from ltx_trainer.p2gs.losses import (
    P2GSLossConfig,
    P2GSLosses,
    build_hdr_pairs,
    hdr_inconsistency_score,
    std_luminance,
)
from ltx_trainer.p2gs.photometric import ViewPhotometricParams, render_ldr, srgb_to_linear, tone_map_gamma
from ltx_trainer.p2gs.pipeline import P2GSConfig, P2GSTrainer, load_checkpoint, render_scene_ldr
from ltx_trainer.p2gs.rasterize import render_linear_hdr

__all__ = [
    "Camera",
    "GaussianModel",
    "P2GSConfig",
    "P2GSLossConfig",
    "P2GSLosses",
    "P2GSTrainer",
    "ViewPhotometricParams",
    "build_hdr_pairs",
    "hdr_inconsistency_score",
    "load_checkpoint",
    "load_scene_cameras",
    "render_ldr",
    "render_linear_hdr",
    "render_scene_ldr",
    "srgb_to_linear",
    "std_luminance",
    "tone_map_gamma",
]
