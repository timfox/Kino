"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.faor.benchmarks import (
    TABLE1_ODI_SR,
    TABLE2_PRIORS,
    faor_beats_osrt_odisr,
    faor_beats_osrt_sun360,
    geodesic_ablation_gain,
)
from ltx_trainer.faor.config import FaorConfig
from ltx_trainer.faor.faor_net import FaorStub


def evaluation_demo_run(scale: float = 2.0) -> dict[str, Any]:
    model = FaorStub(FaorConfig())
    lr = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        out = model(lr, scale=scale)
    return {
        "scale": scale,
        "sr_shape": list(out["sr"].shape),
        "ws_psnr_x8_paper": TABLE1_ODI_SR["FAOR"][8][0],
        "beats_osrt": faor_beats_osrt_odisr(8),
    }


def train_step() -> dict[str, float]:
    cfg = FaorConfig()
    model = FaorStub(cfg)
    lr = torch.randn(1, 3, cfg.lr_size, cfg.lr_size * 2, requires_grad=True)
    hr = model(lr, scale=2.0)["sr"]
    target = torch.randn_like(hr)
    loss = (hr - target).abs().mean()
    loss.backward()
    return {"l1_loss": float(loss.detach())}


def ablation_table_check() -> dict[str, bool]:
    return {
        "faor_beats_osrt_odisr_x8": faor_beats_osrt_odisr(8),
        "faor_beats_osrt_sun360_x8": faor_beats_osrt_sun360(8),
        "geodesic_gain_0p16db": geodesic_ablation_gain() >= 0.15,
        "md_ms_ablation_ordered": TABLE2_PRIORS["FAOR"][8][0]
        >= TABLE2_PRIORS["w/o_Md"][8][0]
        >= TABLE2_PRIORS["w/o_Md_Ms"][8][0],
    }
