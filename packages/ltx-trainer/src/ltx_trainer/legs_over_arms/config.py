"""Legs Over Arms / HST configuration (Le et al. arXiv:2602.09076)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2602.09076"
PAPER_TITLE = (
    "Legs Over Arms: On the Predictive Value of Lower-Body Pose for "
    "Human Trajectory Prediction from Egocentric Robot Perception"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

# Temporal setup (Sec. III-C)
SAMPLE_HZ = 3
HISTORY_STEPS = 6  # 2 s
FUTURE_STEPS = 12  # 4 s
NUM_MODES = 6

# Skeleton (Fig. 2, Sec. III-A)
NUM_KEYPOINTS_3D = 33
NUM_KEYPOINTS_2D = 17
LOWER_BODY_3D = 10
UPPER_BODY_3D = 10
LOWER_BODY_2D = 6

# Training (Sec. III-C)
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


@dataclass
class LegsOverArmsConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    num_agents: int = 4
    history_steps: int = HISTORY_STEPS
    future_steps: int = FUTURE_STEPS
    num_modes: int = NUM_MODES
    feature_config: str = "K3D_L"  # G(s) shorthand
