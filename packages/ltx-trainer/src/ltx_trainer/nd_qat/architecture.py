"""Gohr ND architecture specs and operation accounting (Sec. 1.1, 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.nd_qat.config import (
    CONV0_OUT,
    FEATURE_MAP_SIZE,
    GOHR_OPS,
    INPUT_CHANNELS,
    KERNEL_3X3,
    LIGHTWEIGHT_OPS,
    OPS_RATIO_PCT,
    RESIDUAL_CHANNELS,
)


@dataclass
class LayerOpCount:
    multiplications: int = 0
    additions: int = 0
    boolean: int = 0
    indicators: int = 0

    @property
    def total_gohr(self) -> int:
        return self.multiplications + self.additions

    @property
    def total_lightweight(self) -> int:
        return self.boolean + self.additions + self.indicators


def conv0_gohr_ops() -> LayerOpCount:
    """128 1×1 convs → 128×128 mult, (128-32)×128 add (Sec. 2)."""
    return LayerOpCount(multiplications=128 * 128, additions=(128 - 32) * 128)


def conv0_lightweight_ops() -> LayerOpCount:
    """8 nonzero weights, 4 active output channels (Table 1)."""
    return LayerOpCount(boolean=8 * 128, additions=4 * 128, indicators=4 * 128)


def residual_block_gohr_ops() -> LayerOpCount:
    """Two 3×3 conv layers: 9216 weights each × 128 spatial (Sec. 2)."""
    weights = RESIDUAL_CHANNELS * RESIDUAL_CHANNELS * KERNEL_3X3 * KERNEL_3X3
    mult = weights * FEATURE_MAP_SIZE * 2
    add = (weights - RESIDUAL_CHANNELS) * FEATURE_MAP_SIZE * 2
    return LayerOpCount(multiplications=mult, additions=add)


def residual_block_lightweight_ops() -> LayerOpCount:
    """671 + 2081 nonzero weights across two conv kernels."""
    nz = 671 + 2081
    return LayerOpCount(boolean=352_256, additions=344_064, indicators=8192)


def head_gohr_ops() -> LayerOpCount:
    return LayerOpCount(multiplications=266_240, additions=266_112)


def head_lightweight_ops() -> LayerOpCount:
    return LayerOpCount(boolean=13_877, additions=13_778, indicators=128)


def output_gohr_ops() -> LayerOpCount:
    return LayerOpCount(multiplications=128, additions=126)


def output_lightweight_ops() -> LayerOpCount:
    return LayerOpCount(boolean=64, additions=63, indicators=1)


def total_gohr_from_paper() -> LayerOpCount:
    return LayerOpCount(
        multiplications=GOHR_OPS["multiplications"],
        additions=GOHR_OPS["additions"],
    )


def total_lightweight_from_paper() -> LayerOpCount:
    return LayerOpCount(
        boolean=LIGHTWEIGHT_OPS["boolean"],
        additions=LIGHTWEIGHT_OPS["additions"],
        indicators=LIGHTWEIGHT_OPS["indicators"],
    )


def table1_conv0_weights() -> list[dict[str, Any]]:
    return [
        {"input_channel": 1, "out_0": -1, "out_1": 0, "out_2": 1, "out_3": 0},
        {"input_channel": 15, "out_0": 1, "out_1": 0, "out_2": -1, "out_3": 0},
        {"input_channel": 24, "out_0": 0, "out_1": 1, "out_2": 0, "out_3": -1},
        {"input_channel": 25, "out_0": 0, "out_1": -1, "out_2": 0, "out_3": 1},
    ]


def table5_conv0_boolean() -> list[dict[str, str]]:
    return [
        {"channel": 1, "expression": "C'_l ∧ Cl"},
        {"channel": 15, "expression": "Cl ∧ C'_l"},
        {"channel": 24, "expression": "Cr ∧ C'_r"},
        {"channel": 25, "expression": "C'_r ∧ Cr"},
    ]
