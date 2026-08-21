"""PanoEnv configuration (Lin & Zheng, arXiv:2602.21992)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2602.21992"
PAPER_TITLE = "PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with RL"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/7zk1014/PanoEnv"

# PanoEnv-QA (Sec. 3.1, 4.1)
PANENV_QA_TOTAL = 14_827
PANENV_SCENES = 595
PANENV_ENVIRONMENTS = 60
TEST_SAMPLES = 3_040
ERP_SIZE = (2560, 1280)

# GRPO (Sec. 3.2, Supp. 6.4)
GRPO_GROUP_SIZE = 4
GRPO_KL_BETA = 0.01
REWARD_W_ACC = 0.9
REWARD_W_FMT = 0.1
LORA_RANK = 16
LORA_ALPHA = 32
BASE_MODEL = "Qwen2.5-VL-7B-Instruct"
LEARNING_RATE = 5e-6
TRAIN_EPOCHS = 2


@dataclass
class PanoEnvConfig:
    height: int = 64
    width: int = 128
    embed_dim: int = 64
    grpo_k: int = GRPO_GROUP_SIZE
    kl_beta: float = GRPO_KL_BETA
    lora_rank: int = LORA_RANK
    curriculum_stage: int = 1
