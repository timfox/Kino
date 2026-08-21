"""FEFormer training and reporting pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.feformer.config import FEFormerConfig
from ltx_trainer.feformer.losses import segmentation_loss
from ltx_trainer.feformer.metrics import (
    table2_amos_mean,
    table3_hepatic_flare,
    table4_brain_tumor,
    table5_complexity,
    table6_ablation_modules,
    table7_fdsa_fgmlp_ablation,
    table8_fusion_comparison,
    table9_flare_generalization,
)
from ltx_trainer.feformer.model import FEFormer


def train_step(model: FEFormer, x: Tensor, y: Tensor) -> tuple[Tensor, dict[str, float]]:
    pred = model(x)
    if pred.shape[-3:] != y.shape[-3:]:
        pred = torch.nn.functional.interpolate(pred, size=y.shape[-3:], mode="trilinear", align_corners=False)
    loss = segmentation_loss(pred, y)
    with torch.no_grad():
        acc = float((pred.argmax(1) == y).float().mean())
    return loss, {"loss_seg": float(loss), "pixel_acc": acc}


def count_parameters(model: FEFormer) -> int:
    return sum(p.numel() for p in model.parameters())


def paper_report() -> dict[str, Any]:
    return {
        "amos": table2_amos_mean(),
        "hepatic_flare": table3_hepatic_flare(),
        "brain_tumor": table4_brain_tumor(),
        "complexity": table5_complexity(),
        "ablation_modules": table6_ablation_modules(),
        "ablation_fdsa_fgmlp": table7_fdsa_fgmlp_ablation(),
        "fusion_comparison": table8_fusion_comparison(),
        "flare_generalization": table9_flare_generalization(),
    }


def ablation_configs() -> list[dict[str, bool]]:
    rows = table6_ablation_modules()
    return [{"fdsa": r["fdsa"], "fgmlp": r["fgmlp"], "waff": r["waff"], "fcsb": r["fcsb"]} for r in rows]
