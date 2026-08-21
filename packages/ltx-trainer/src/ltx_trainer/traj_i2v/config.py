"""Configuration for trajectory-guided maritime I2V reconstruction (arXiv:2605.16420)."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2605.16420"
PAPER_TITLE = (
    "Video Reconstruction using Diffusion-based Image-to-Video Generation with Trajectory Guidance"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
SG_I2V_REF = "https://arxiv.org/abs/2411.04989"

# MUSIT / maritime ASV clip defaults (Section III-B)
DEFAULT_VESSEL_IDS = (99999, 100000)
DEFAULT_VESSEL_COLORS = {99999: "green", 100000: "yellow"}
DEFAULT_LOG_TIME_OFFSET_S = 21.0
DEFAULT_CAMERA_YAW_DEG = 100.0
DEFAULT_FPS = 7.0
DEFAULT_NUM_GENERATED_FRAMES = 14
DEFAULT_POST_WIDTH = 1024
DEFAULT_POST_HEIGHT = 576

MLAT_M_PER_DEG = 111_320.0


@dataclass
class TrajI2VConfig:
    """Pipeline hyper-parameters matching the paper's two-second ASV clip setup."""

    num_frames: int = DEFAULT_NUM_GENERATED_FRAMES
    fps: float = DEFAULT_FPS
    camera_yaw_deg: float = DEFAULT_CAMERA_YAW_DEG
    log_time_offset_s: float = DEFAULT_LOG_TIME_OFFSET_S
    corner_box_size: int = 35
    corner_inset_px: int = 30
    image_width: int = DEFAULT_POST_WIDTH
    image_height: int = DEFAULT_POST_HEIGHT
    # SG-I2V guidance (ablation-selected in paper; stub defaults)
    latent_opt_iters: int = 50
    heatmap_sigma: float = 8.0
    fft_blend_ratio: float = 0.5
    vessel_ids: tuple[int, ...] = DEFAULT_VESSEL_IDS
    # Optional external SG-I2V checkpoint / API (not bundled in GOPEX)
    sg_i2v_backend: str = "stub"
    draw_trajectory_arrows: bool = True
