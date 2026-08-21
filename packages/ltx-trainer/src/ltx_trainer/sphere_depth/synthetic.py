"""Synthetic ERP + landmark batches for smoke evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.sphere_depth.config import SphereDepthConfig


def synthetic_scene(
    cfg: SphereDepthConfig,
    *,
    num_landmarks: int = 20,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else None
    h, w = cfg.erp_height, cfg.erp_width
    erp = torch.rand(1, 3, h, w, device=dev)
    u = torch.randint(0, w, (num_landmarks,), device=dev, dtype=torch.float32)
    v = torch.randint(0, h, (num_landmarks,), device=dev, dtype=torch.float32)
    gt_depth = 1.0 + 3.0 * torch.rand(num_landmarks, device=dev)
    return {"erp": erp, "landmark_u": u, "landmark_v": v, "gt_depth": gt_depth}
