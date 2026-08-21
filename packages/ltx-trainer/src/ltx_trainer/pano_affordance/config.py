"""PanoAffordanceNet configuration (Zhu et al. arXiv:2603.09760)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2603.09760"
PAPER_TITLE = (
    "PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/GL-ZHU925/PanoAffordanceNet"

# 360-AGD (Sec. IV)
NUM_AFFORDANCE_CLASSES = 19
INPUT_SIZE = (560, 1120)  # H, W
LORA_RANK = 16
OSDH_TOP_K = 10

# Training (Sec. V-A)
TRAIN_ITERATIONS = 20_000
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
LAMBDA_BCE = 1.0
LAMBDA_KL = 1.0
LAMBDA_RTC = 1.0


@dataclass
class PanoAffordanceConfig:
    height: int = 56
    width: int = 112
    num_classes: int = NUM_AFFORDANCE_CLASSES
    embed_dim: int = 64
    lora_rank: int = LORA_RANK
    osdh_top_k: int = OSDH_TOP_K
    lambda_bce: float = LAMBDA_BCE
    lambda_kl: float = LAMBDA_KL
    lambda_rtc: float = LAMBDA_RTC
