"""Paper metrics and Table 4 comparison."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nd_qat.config import (
    ACCURACY_DROP_PCT,
    CONV0_ONLY_ACCURACY_DROP_PCT,
    GOHR_OPS,
    LIGHTWEIGHT_OPS,
    OPS_RATIO_PCT,
)
from ltx_trainer.nd_qat.architecture import table1_conv0_weights, table5_conv0_boolean


def table4_comparison() -> list[dict[str, Any]]:
    return [
        {
            "model": "Gohr Distinguisher",
            "multiplications_or_boolean": GOHR_OPS["multiplications"],
            "additions": GOHR_OPS["additions"],
            "indicators": GOHR_OPS["indicators"],
            "accuracy_pct": GOHR_OPS["accuracy_pct"],
        },
        {
            "model": "Lightweight Distinguisher",
            "multiplications_or_boolean": LIGHTWEIGHT_OPS["boolean"],
            "additions": LIGHTWEIGHT_OPS["additions"],
            "indicators": LIGHTWEIGHT_OPS["indicators"],
            "accuracy_pct": LIGHTWEIGHT_OPS["accuracy_pct"],
        },
    ]


def table6_conv0_only() -> dict[str, float]:
    return {
        "after_lightweight_pct": 94.64,
        "before_lightweight_pct": 94.95,
        "accuracy_loss_pct": CONV0_ONLY_ACCURACY_DROP_PCT,
    }


def table2_conv1_sample() -> list[dict[str, str]]:
    return [
        {
            "channel": 0,
            "equation": "Y = I(Sum1 > Sum2); Sum1 = H1+I1+H15+I15+...; Sum2 = B1+D1+E1+...",
        },
        {"channel": 1, "equation": "Y = I(Sum1 > Sum2); partial boolean sums over ±1 weights"},
    ]


def table3_output_weights_sample() -> list[dict[str, int]]:
    return [
        {"input": 0, "out0": -1, "out1": -1, "out2": 1, "out3": 1},
        {"input": 1, "out0": 1, "out1": 1, "out2": -1, "out3": -1},
    ]


def ops_reduction_ratio() -> float:
    gohr_total = GOHR_OPS["multiplications"] + GOHR_OPS["additions"]
    lw_total = LIGHTWEIGHT_OPS["boolean"] + LIGHTWEIGHT_OPS["additions"] + LIGHTWEIGHT_OPS["indicators"]
    return lw_total / gohr_total
