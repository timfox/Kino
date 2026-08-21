"""Training and evaluation pipeline for VUGA."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.vuga.config import VUGAConfig
from ltx_trainer.vuga.losses import mse_quality_loss, plcc_srcc_stub
from ltx_trainer.vuga.synthetic import synthetic_batch
from ltx_trainer.vuga.vuga_net import VUGA


def train_step(
    model: VUGA,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    out = model(batch["rgb"])
    loss = mse_quality_loss(out["score"], batch["mos"])
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    metrics = {"loss": float(loss.item())}
    with torch.no_grad():
        metrics.update(plcc_srcc_stub(out["score"], batch["mos"]))
    return metrics


def evaluation_demo_run(cfg: VUGAConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or VUGAConfig(input_size=64, stage_dims=(32, 64, 128, 256), cmp_dim=128)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, batch_size=4, device=dev)
    model = VUGA(cfg).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.train_lr, weight_decay=cfg.weight_decay)
    m1 = train_step(model, batch, optimizer=opt)
    with torch.no_grad():
        out = model(batch["rgb"])
    return {
        "device": str(dev),
        "train": m1,
        "score_mean": float(out["score"].mean().item()),
        "trainable_params": sum(p.numel() for p in trainable),
    }
