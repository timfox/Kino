"""Panorama-Language Model configuration (Fan et al. arXiv:2603.09573)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2603.09573"
PAPER_TITLE = "More than the Sum: Panorama-Language Models for Adverse Omni-Scenes"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
CODE_URL = "https://github.com/InSAI-Lab/PanoVQA"

# PanoVQA (Sec. III-A, IV)
PANOVQA_TOTAL_QA = 653_000
PANOVQA_TRAIN = 538_635
PANOVQA_VAL = 115_279
PANOVQA_MINI_TRAIN = 17_378
PANOVQA_MINI_VAL = 8_184
PANOVQA_FRAMES = 44_600
NUM_QA_TYPES = 12
INPUT_SIZE = (560, 1120)

# PSA hyperparameters (Sec. 4.3, Fig. 5)
PSA_TOP_K = 512
PSA_BOTTLENECK_DIM = 196
BASE_MODEL = "Qwen2.5-VL-7B-Instruct"

# Training (Sec. 4.1)
TRAIN_ITERATIONS = 20_000
BATCH_SIZE = 4
LEARNING_RATE = 5e-6
LORA_RANK = 16


@dataclass
class PanoLMConfig:
    height: int = 56
    width: int = 112
    embed_dim: int = 64
    num_heads: int = 4
    window_size: int = 8
    psa_top_k: int = PSA_TOP_K
    psa_bottleneck: int = PSA_BOTTLENECK_DIM
    lora_rank: int = LORA_RANK
