"""Synthetic ERP batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.mtpano.auxiliary import build_auxiliary_targets
from ltx_trainer.mtpano.config import MTPanoConfig


def synthetic_batch(
    cfg: MTPanoConfig,
    *,
    batch_size: int = 2,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    b, h, w = batch_size, cfg.height, cfg.width
    image = torch.rand(b, 3, h, w, device=dev)
    sem = torch.randint(0, 19, (b, h, w), device=dev)
    depth = torch.rand(b, 1, h, w, device=dev) * 5 + 0.5
    normals = torch.randn(b, 3, h, w, device=dev)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-6)
    aux = build_auxiliary_targets(image, depth)
    return {
        "image": image,
        "semseg": sem,
        "depth": depth,
        "normals": normals,
        "grad": aux["grad"],
        "edf": aux["edf"],
        "point_map": aux["point_map"],
    }
