"""MTPano training smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.mtpano.benchmarks import table1_ours
from ltx_trainer.mtpano.config import MTPanoConfig
from ltx_trainer.mtpano.losses import total_loss
from ltx_trainer.mtpano.pd_bridgenet import PDBridgeNetStub
from ltx_trainer.mtpano.synthetic import synthetic_batch


def train_step(
    model: PDBridgeNetStub,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    preds = model(batch["image"])
    targets = {
        "semseg": batch["semseg"],
        "depth": batch["depth"],
        "normals": batch["normals"],
        "grad": batch["grad"],
        "edf": batch["edf"],
        "point_map": batch["point_map"],
    }
    loss = total_loss(preds, targets)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {"loss": float(loss.item())}


def evaluation_demo_run(cfg: MTPanoConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or MTPanoConfig(height=128, width=256)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=1, device=dev)
    model = PDBridgeNetStub(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    m = train_step(model, batch, optimizer=opt)
    ref = table1_ours()
    return {
        "device": str(dev),
        "train": m,
        "ref_mIoU_structured3d": ref["mIoU"],
        "ref_AbsRel": ref["AbsRel"],
    }
