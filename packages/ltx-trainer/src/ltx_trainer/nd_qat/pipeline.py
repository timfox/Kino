"""ND-QAT training pipeline and reporting."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.nd_qat.architecture import (
    table1_conv0_weights,
    table5_conv0_boolean,
    total_gohr_from_paper,
    total_lightweight_from_paper,
)
from ltx_trainer.nd_qat.config import NDQATConfig
from ltx_trainer.nd_qat.metrics import (
    ops_reduction_ratio,
    table2_conv1_sample,
    table3_output_weights_sample,
    table4_comparison,
    table6_conv0_only,
)
from ltx_trainer.nd_qat.model import GohrDistinguisher, LightweightDistinguisher


def train_step(model: GohrDistinguisher | LightweightDistinguisher, x: Tensor, y: Tensor) -> tuple[Tensor, dict[str, float]]:
    pred = model(x)
    loss = F.binary_cross_entropy(pred, y)
    with torch.no_grad():
        acc = ((pred >= 0.505) == y.bool()).float().mean()
    return loss, {"loss": float(loss.detach()), "accuracy": float(acc)}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def paper_report() -> dict[str, Any]:
    gohr = total_gohr_from_paper()
    lw = total_lightweight_from_paper()
    return {
        "table4_comparison": table4_comparison(),
        "table6_conv0_only": table6_conv0_only(),
        "conv0_weights": table1_conv0_weights(),
        "conv0_boolean": table5_conv0_boolean(),
        "conv1_sample": table2_conv1_sample(),
        "output_weights_sample": table3_output_weights_sample(),
        "ops_ratio_pct": ops_reduction_ratio() * 100,
        "gohr_total_ops": gohr.total_gohr,
        "lightweight_total_ops": lw.total_lightweight,
    }


def knowledge() -> dict[str, Any]:
    return {
        "name": "QAT Lightweight Neural Distinguisher",
        "url": "https://arxiv.org/abs/2603.05791",
        "doi": "10.48550/arXiv.2603.05791",
        "cipher": "SPECK32/64",
        "quantization": "LSQ ternary 1.58-bit {0, ±1}",
        "ops_reduction_pct": 13.9,
        "accuracy_drop_pct": 2.87,
    }
