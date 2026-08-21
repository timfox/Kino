"""FastViDAR — real-time omnidirectional depth via AHA (arXiv:2509.23733)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2509.23733"
PAPER_TITLE = "FastViDAR: Real-Time Omnidirectional Depth Estimation via Alternative Hierarchical Attention"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://3f7dfc.github.io/FastVidar/"

# Default eval / train setup (Sec. IV)
ERP_HEIGHT = 320
ERP_WIDTH = 640
NUM_CAMERAS = 4
WINDOW_PH = 7
WINDOW_PW = 7
HUBER_DELTA = 1.0
GRADIENT_LOSS_WEIGHT = 1.0


@dataclass
class FastViDARConfig:
    height: int = ERP_HEIGHT
    width: int = ERP_WIDTH
    num_frames: int = NUM_CAMERAS
    window_h: int = WINDOW_PH
    window_w: int = WINDOW_PW
    hidden_dim: int = 64
    num_aha_blocks: int = 2
    num_refine_layers: int = 2
    use_global_attention: bool = True
    huber_delta: float = HUBER_DELTA
    lambda_grad: float = GRADIENT_LOSS_WEIGHT
