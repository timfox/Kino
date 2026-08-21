"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.sdt2i.benchmarks import (
    TABLE3_MAIN,
    bg_eppa_improves_mpf,
    mstd_beats_mpf_iou,
    mstd_matches_md_baseline,
)
from ltx_trainer.sdt2i.config import Sdt2iConfig
from ltx_trainer.sdt2i.masks import erp_mask_from_perspective
from ltx_trainer.sdt2i.sdt2i_net import Sdt2iStub


def evaluation_demo_run(variant: str = "mstd") -> dict[str, Any]:
    cfg = Sdt2iConfig()
    model = Sdt2iStub(cfg, variant=variant)
    latent = torch.randn(1, 4, 32, 64)
    m1 = erp_mask_from_perspective(0.5, 0.0, erp_h=32, erp_w=64)
    m2 = erp_mask_from_perspective(-1.0, 0.2, erp_h=32, erp_w=64)
    with torch.no_grad():
        out = model(latent, [m1, m2], steps=3)
    return {
        "variant": variant,
        "latent_shape": list(out["latent"].shape),
        "mstd_iou_paper": TABLE3_MAIN["MSTD"]["iou"],
        "beats_mpf": mstd_beats_mpf_iou(),
    }


def train_step() -> dict[str, float]:
    model = Sdt2iStub(Sdt2iConfig(), variant="mstd")
    latent = torch.randn(1, 4, 24, 48, requires_grad=True)
    m = erp_mask_from_perspective(0.0, 0.0, erp_h=24, erp_w=48)
    out = model(latent, [m], steps=2)
    loss = out["latent"].pow(2).mean() + out["certainty"].mean()
    loss.backward()
    return {"loss": float(loss.detach())}


def ablation_table_check() -> dict[str, bool]:
    return {
        "mstd_beats_mpf_iou": mstd_beats_mpf_iou(),
        "mstd_near_md_pano_lora": mstd_matches_md_baseline(),
        "bg_eppa_improves_mpf": bg_eppa_improves_mpf(),
        "mpf_pers_fails": TABLE3_MAIN["MPF_md_pers"]["iou"] < 0.1,
    }
