"""Synthetic 360-AGD batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.pano_affordance.config import PanoAffordanceConfig


def synthetic_batch(
    cfg: PanoAffordanceConfig,
    batch_size: int = 2,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    h, w = cfg.height, cfg.width
    image = torch.rand(batch_size, 3, h, w, device=dev)
    # patch grid from vision encoder stride
    l = (h // 14) * (w // 14)
    if l <= 0:
        l = 32
    target = torch.zeros(batch_size, cfg.num_classes, l, device=dev)
    for b in range(batch_size):
        for c in range(min(3, cfg.num_classes)):
            idx = torch.randint(0, l, (4,), device=dev)
            target[b, c, idx] = 1.0
    target = torch.sigmoid(target * 4.0)
    return {"image": image, "heatmap_gt": target}
