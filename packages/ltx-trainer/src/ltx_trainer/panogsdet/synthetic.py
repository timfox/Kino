"""Synthetic Structured3D-style panorama batch for smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig


def synthetic_batch(
    cfg: PanoGSDetConfig | None = None,
    *,
    batch: int = 1,
    device: torch.device | None = None,
) -> dict[str, Tensor | list[Tensor]]:
    cfg = cfg or PanoGSDetConfig()
    h, w = cfg.height, cfg.width
    dev = device or torch.device("cpu")
    rgb = torch.rand(batch, 3, h, w, device=dev)
    depth_gt = torch.rand(batch, 1, h, w, device=dev) * 3.0 + 1.0
    sem_gt = torch.randn(batch, cfg.num_classes, h, w, device=dev)
    # sparse GT boxes [center(3), size(3), yaw(1)]
    gt_boxes: list[Tensor] = []
    gt_scores: list[Tensor] = []
    for _ in range(batch):
        n = 4
        centers = torch.tensor(
            [[0.5, 1.2, 2.0], [-0.8, 0.9, 1.5], [1.1, 1.0, -0.6], [0.0, 1.5, 0.3]], device=dev
        )
        sizes = torch.tensor([[1.0, 0.5, 2.0], [0.6, 0.6, 0.6], [1.2, 0.8, 0.4], [0.9, 0.4, 1.1]], device=dev)
        yaw = torch.zeros(n, 1, device=dev)
        gt_boxes.append(torch.cat([centers, sizes, yaw], dim=-1))
        gt_scores.append(torch.ones(n, device=dev))
    return {
        "rgb": rgb,
        "depth_gt": depth_gt,
        "sem_gt": sem_gt,
        "gt_boxes": gt_boxes,
        "gt_scores": gt_scores,
    }
