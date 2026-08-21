"""LoRA-style latent flow adapter stub (AVControl + flow matching, Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LumiVidLoRAStub(nn.Module):
    """Predict velocity field on HDR target latents conditioned on degraded SDR reference."""

    def __init__(self, latent_ch: int = 16, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Conv2d(latent_ch * 2, rank, 1)
        self.up = nn.Conv2d(rank, latent_ch, 1)

    def forward(self, z_noisy: Tensor, z_ref: Tensor, t: Tensor) -> Tensor:
        if z_ref.shape[-2:] != z_noisy.shape[-2:]:
            z_ref = torch.nn.functional.interpolate(z_ref, size=z_noisy.shape[-2:], mode="bilinear", align_corners=False)
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, z_noisy.shape[-2], z_noisy.shape[-1])
        x = torch.cat([z_noisy, z_ref], dim=1)
        return self.up(torch.relu(self.down(x))) + 0.01 * t_map


def flow_matching_loss(model: LumiVidLoRAStub, z_tgt: Tensor, z_ref: Tensor) -> Tensor:
    """Linear flow matching: predict z_tgt - z0 from interpolated state."""
    z0 = torch.randn_like(z_tgt)
    t = torch.rand(z_tgt.shape[0], device=z_tgt.device, dtype=z_tgt.dtype)
    t_b = t.view(-1, 1, 1, 1)
    z_t = (1.0 - t_b) * z0 + t_b * z_tgt
    target_v = z_tgt - z0
    pred_v = model(z_t, z_ref, t)
    return torch.nn.functional.mse_loss(pred_v, target_v)
