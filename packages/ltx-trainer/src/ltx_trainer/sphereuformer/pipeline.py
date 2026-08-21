"""Train / eval demo stubs."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.sphereuformer.benchmarks import (
    TABLE4_RANK7,
    beats_elite360d_seg_s2d3d,
    beats_panoformer_depth_s2d3d,
    rank8_beats_panoformer_mae,
)
from ltx_trainer.sphereuformer.config import SphereUFormerConfig
from ltx_trainer.sphereuformer.losses import berhu_loss
from ltx_trainer.sphereuformer.sphereuformer_net import SphereUFormerStub


def train_step(*, batch_size: int = 2, h: int = 32, w: int = 64) -> dict[str, Any]:
    cfg = SphereUFormerConfig(rank=7, task="depth")
    model = SphereUFormerStub(cfg)
    erp = torch.randn(batch_size, 3, h, w)
    target = torch.rand(batch_size, h, w) * 5.0
    out = model(erp)
    loss = berhu_loss(out["depth"], target)
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "nodes": int(cfg.rank),
        "task": cfg.task,
    }


def evaluation_demo_run() -> dict[str, Any]:
    model = SphereUFormerStub(SphereUFormerConfig(rank=7))
    erp = torch.randn(1, 3, 64, 128)
    out = model(erp)
    return {
        "depth_shape": list(out["depth"].shape),
        "reference_mae": TABLE4_RANK7["OURS"]["s2d3d"]["mae"],
    }


def ablation_table_check() -> dict[str, bool]:
    return {
        "beats_panoformer_depth_s2d3d": beats_panoformer_depth_s2d3d(),
        "beats_elite360d_seg_s2d3d": beats_elite360d_seg_s2d3d(),
        "rank8_beats_panoformer_mae": rank8_beats_panoformer_mae(),
    }
