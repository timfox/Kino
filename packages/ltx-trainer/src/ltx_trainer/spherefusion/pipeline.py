"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.spherefusion.benchmarks import (
    TABLE1_S2D3D,
    beats_sphere_depth_s2d3d,
    faster_than_unifuse_s2d3d,
    gatefuse_best_ablation,
)
from ltx_trainer.spherefusion.config import SphereFusionConfig
from ltx_trainer.spherefusion.losses import berhu_loss
from ltx_trainer.spherefusion.spherefusion_net import SphereFusionStub


def evaluation_demo_run() -> dict[str, Any]:
    model = SphereFusionStub(SphereFusionConfig())
    erp = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        out = model(erp)
    return {
        "depth_shape": list(out["depth"].shape),
        "s2d3d_mre": TABLE1_S2D3D["SphereFusion"]["mre"],
        "beats_sphere_depth": beats_sphere_depth_s2d3d(),
    }


def train_step() -> dict[str, float]:
    model = SphereFusionStub(SphereFusionConfig())
    erp = torch.randn(1, 3, 32, 64, requires_grad=True)
    gt = torch.rand(1, 32, 64) * 5 + 0.5
    pred = model(erp)["depth"]
    loss = berhu_loss(pred, gt)
    loss.backward()
    return {"berhu_loss": float(loss.detach())}


def ablation_table_check() -> dict[str, bool]:
    return {
        "beats_sphere_depth_s2d3d": beats_sphere_depth_s2d3d(),
        "faster_than_unifuse": faster_than_unifuse_s2d3d(),
        "gatefuse_best_rmse": gatefuse_best_ablation(),
        "s2d3d_lowest_mre_among_listed": TABLE1_S2D3D["SphereFusion"]["mre"]
        < TABLE1_S2D3D["UniFuse"]["mre"],
    }
