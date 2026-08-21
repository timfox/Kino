"""CLRNet configuration (Kegl et al. arXiv:2603.15767)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2603.15767"
PAPER_TITLE = (
    "CLRNet: Targetless Extrinsic Calibration for Camera, Lidar and 4D Radar Using Deep Learning"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/tudelft-iv"

# VoD splits (Sec. IV-A)
VOD_TRAIN_FRAMES = 25_000
VOD_VAL_FRAMES = 3_300
VOD_TEST_FRAMES = 1_335
DUAL_RADAR_FRAMES = 10_007

# Input geometry (Sec. III-B, IV-B)
ERP_NATIVE = (1024, 2048)  # H, W before resize
ENCODER_SIZE = (512, 1024)  # H, W
LIDAR_CHANNELS = 2  # range, intensity
RADAR_CHANNELS = 4  # range, RCS, velocity, time
CORRELATION_SIZE = (32, 64)  # H, W feature maps for matching

# Training (Sec. IV-B)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LAMBDA_LOOP = 0.25
LAMBDA_R = 1.0
LAMBDA_T = 2.0
LAMBDA_PAIRWISE = 0.5
CLRNET_PLUS4_FRAMES = 4


@dataclass
class CLRNetConfig:
    height: int = 64
    width: int = 128
    lidar_channels: int = LIDAR_CHANNELS
    radar_channels: int = RADAR_CHANNELS
    feature_dim: int = 64
    correlation_hw: tuple[int, int] = (16, 32)
    lambda_loop: float = LAMBDA_LOOP
    lambda_pairwise: float = LAMBDA_PAIRWISE
    lambda_r: float = LAMBDA_R
    lambda_t: float = LAMBDA_T
    use_depth_branch: bool = True
    use_equirect: bool = True
    num_frames: int = 1  # CLRNet+4 sets to 4
