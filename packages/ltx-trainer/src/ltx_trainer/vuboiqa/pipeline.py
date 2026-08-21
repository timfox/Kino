"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.vuboiqa.benchmarks import (
    TABLE1_OIQA_MAIN,
    cross_db_best_srcc,
    ours_beats_assessor360_jufe,
    ours_lower_complexity_than_assessor360,
    pdff_improves_over_backbone,
)
from ltx_trainer.vuboiqa.config import VuBoiqaConfig
from ltx_trainer.vuboiqa.metrics import pearson, spearman_proxy
from ltx_trainer.vuboiqa.vuboiqa_net import VuBoiqaStub


def evaluation_demo_run(cfg: VuBoiqaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VuBoiqaConfig(num_patches=4, patch_size=64)
    model = VuBoiqaStub(cfg)
    erp = torch.rand(1, 3, 128, 256)
    mos = torch.tensor([3.5])
    with torch.no_grad():
        out = model(erp, mos=mos)
    return {
        "quality_shape": list(out["quality"].shape),
        "loss": float(out["loss"].item()),
        "jufe_srcc_paper": TABLE1_OIQA_MAIN["VU-BOIQA"]["jufe_srcc"],
        "beats_assessor360_jufe": ours_beats_assessor360_jufe(),
        "lower_flops": ours_lower_complexity_than_assessor360(),
    }


def train_step(cfg: VuBoiqaConfig | None = None) -> dict[str, float]:
    cfg = cfg or VuBoiqaConfig(num_patches=3, patch_size=48)
    model = VuBoiqaStub(cfg)
    erp = torch.rand(2, 3, 96, 192)
    mos = torch.tensor([2.1, 4.3])
    out = model(erp, mos=mos)
    out["loss"].backward()
    return {"loss": float(out["loss"].detach())}


def correlation_demo() -> dict[str, float]:
    pred = torch.linspace(1, 5, 20)
    tgt = pred + 0.2 * torch.randn(20)
    return {
        "plcc": float(pearson(pred, tgt).item()),
        "srcc": float(spearman_proxy(pred, tgt).item()),
    }


def ablation_table_check() -> dict[str, bool]:
    return {
        "beats_assessor360_jufe": ours_beats_assessor360_jufe(),
        "lower_complexity": ours_lower_complexity_than_assessor360(),
        "pdff_helps": pdff_improves_over_backbone(),
        "cross_db_best_srcc": cross_db_best_srcc(),
    }
