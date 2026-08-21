"""Synthetic NFoV + ERP pairs for smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gimbal360.config import Gimbal360Config


def synthetic_batch(
    cfg: Gimbal360Config,
    batch_size: int = 2,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else None
    erp = torch.rand(batch_size, 3, cfg.erp_height, cfg.erp_width, device=dev)
    perspective = torch.rand(batch_size, 3, cfg.perspective_height, cfg.perspective_width, device=dev)
    mask = torch.zeros(batch_size, 1, cfg.erp_height // 4, cfg.erp_width // 4, device=dev)
    mask[..., : mask.shape[-1] // 3] = 1.0
    z_ref = torch.randn(batch_size, cfg.latent_channels, cfg.erp_height // 4, cfg.erp_width // 4, device=dev)
    flow_gt = torch.randn(batch_size, 2, cfg.perspective_height, cfg.perspective_width, device=dev) * 0.05
    noise = torch.randn_like(z_ref)
    return {
        "erp": erp,
        "perspective": perspective,
        "mask": mask,
        "z_ref": z_ref,
        "flow_gt": flow_gt,
        "noise": noise,
    }
