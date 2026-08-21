"""Training and evaluation pipeline for PanoGSDet."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig
from ltx_trainer.panogsdet.cubemap import erp_semantic_to_cubemap
from ltx_trainer.panogsdet.losses import (
    detection_confidence_loss,
    detection_regression_loss,
    semantic_cubemap_loss,
    total_loss,
)
from ltx_trainer.panogsdet.panogsdet_net import PanoGSDet
from ltx_trainer.panogsdet.synthetic import synthetic_batch


def train_step(
    model: PanoGSDet,
    batch: dict[str, Tensor | list[Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    face_size: int = 32,
    max_gaussians: int = 512,
) -> dict[str, float]:
    model.train()
    rgb = batch["rgb"]
    assert isinstance(rgb, Tensor)
    out = model(rgb, face_size=face_size, max_gaussians=max_gaussians)
    sem_gt = batch["sem_gt"]
    assert isinstance(sem_gt, Tensor)
    target_cube = erp_semantic_to_cubemap(torch.sigmoid(sem_gt), face_size=face_size)
    l_sem = semantic_cubemap_loss(out["cubemap"], target_cube)
    det = out["detection"]
    gt_boxes = batch["gt_boxes"]
    gt_scores = batch["gt_scores"]
    assert isinstance(gt_boxes, list) and isinstance(gt_scores, list)
    l_reg = detection_regression_loss(det["boxes"], gt_boxes)
    l_conf = detection_confidence_loss(det["scores"], gt_scores)
    loss = total_loss(l_sem, l_reg, l_conf)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            model.cfg.grad_clip,
        )
        optimizer.step()
    n_boxes = sum(int(b.shape[0]) for b in det["boxes"])
    return {
        "loss": float(loss.item()),
        "l_sem": float(l_sem.item()),
        "l_reg": float(l_reg.item()),
        "l_conf": float(l_conf.item()),
        "num_proposals": n_boxes,
    }


def evaluation_demo_run(
    cfg: PanoGSDetConfig | None = None,
    *,
    device: str = "cpu",
    face_size: int = 24,
    max_gaussians: int = 256,
) -> dict[str, Any]:
    cfg = cfg or PanoGSDetConfig(height=64, width=128, feature_dim=16, opt_blocks=1)
    dev = torch.device(device)
    batch = synthetic_batch(cfg, device=dev)
    model = PanoGSDet(cfg).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train_lr, weight_decay=cfg.train_weight_decay)
    m1 = train_step(model, batch, optimizer=opt, face_size=face_size, max_gaussians=max_gaussians)
    with torch.no_grad():
        out = model(batch["rgb"], face_size=face_size, max_gaussians=max_gaussians)
    return {
        "device": str(dev),
        "train": m1,
        "num_gaussians": out["gaussians"].num_gaussians,
        "depth_mean": float(out["depth"].mean().item()),
        "proposal_count": sum(int(b.shape[0]) for b in out["detection"]["boxes"]),
    }
