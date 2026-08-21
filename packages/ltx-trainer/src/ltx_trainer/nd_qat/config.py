"""ND-QAT configuration (Xiong et al., arXiv:2603.05791)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2603.05791"
PAPER_DOI = "10.48550/arXiv.2603.05791"
PAPER_TITLE = (
    "A Quantization-Aware Training Based Lightweight Method for Neural Distinguishers"
)

# SPECK32/64 experiment (Sec. 2)
SPECK_ROUNDS = 6
INPUT_DIFF_HEX = "0x0040/0000"
SAMPLES_PER_CLASS = 5_000_000
THRESHOLD = 0.505

# LSQ ternary quantization (Sec. 1.2.1)
QUANT_BITS_EQUIVALENT = 1.58
WEIGHT_VALUES = (-1, 0, 1)

# Table 4 operation counts and accuracy
GOHR_OPS = {
    "multiplications": 2_642_048,
    "additions": 2_629_630,
    "indicators": 0,
    "accuracy_pct": 94.95,
}
LIGHTWEIGHT_OPS = {
    "boolean": 367_221,
    "additions": 358_417,
    "indicators": 8_833,
    "accuracy_pct": 92.21,
}
OPS_RATIO_PCT = 13.9
ACCURACY_DROP_PCT = 2.87
CONV0_ONLY_ACCURACY_DROP_PCT = 0.3

# Architecture dimensions (Gohr ND, Fig. 1)
INPUT_CHANNELS = 4
INPUT_HEIGHT = 16
GROUP_SIZE = 8  # 2D extension (Sec. 2)
CONV0_OUT = 32
RESIDUAL_CHANNELS = 32
KERNEL_3X3 = 3
FEATURE_MAP_SIZE = 128  # 16 * 8 after reshape for op counting


@dataclass
class NDQATConfig:
    input_channels: int = INPUT_CHANNELS
    input_height: int = INPUT_HEIGHT
    group_size: int = GROUP_SIZE
    conv0_out: int = CONV0_OUT
    residual_channels: int = RESIDUAL_CHANNELS
    quant_bits: float = QUANT_BITS_EQUIVALENT
    threshold: float = THRESHOLD
