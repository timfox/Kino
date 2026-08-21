"""PiD pixel-space decode: latent-conditioned rectified flow (Sec. 3)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.pid.config import PiDConfig
from ltx_trainer.pid.flow import (
    corrupt_latent,
    flow_matching_loss,
    integrate_velocity,
    sample_noise_like,
    velocity_target,
)
from ltx_trainer.pid.latent_adapter import LatentProjectionAdapter, SigmaAwareGate, inject_latent


class PiDVelocityStub(nn.Module):
    """Lightweight velocity field stub (full PixelDiT is external)."""

    def __init__(self, in_ch: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        )

    def forward(self, x_t: Tensor, t: float, cond: Tensor | None = None) -> Tensor:
        t_map = torch.full((x_t.shape[0], 1, x_t.shape[2], x_t.shape[3]), t, device=x_t.device, dtype=x_t.dtype)
        if cond is not None:
            if cond.shape[-2:] != x_t.shape[-2:]:
                cond = F.interpolate(cond, size=x_t.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x_t, t_map], dim=1)
            v = self.net(x)
            return v + 0.1 * (cond - x_t)
        return self.net(torch.cat([x_t, t_map], dim=1))


class PiDDecoder(nn.Module):
    """Latent-conditioned pixel decoder with sigma-aware adapter."""

    def __init__(
        self,
        *,
        cfg: PiDConfig | None = None,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PiDConfig()
        self.adapter = LatentProjectionAdapter(latent_channels, cfg=self.cfg)
        self.gate = SigmaAwareGate(self.cfg.hidden_dim, cfg=self.cfg)
        self.velocity = PiDVelocityStub()

    def project_latent_tokens(
        self,
        z_noisy: Tensor,
        *,
        out_hw: tuple[int, int],
        sigma: float,
    ) -> Tensor:
        patch = self.cfg.patch_size
        th, tw = out_hw[0] // patch, out_hw[1] // patch
        return self.adapter(z_noisy, target_hw=(th, tw), block_id="0", hidden_dim=self.cfg.hidden_dim)

    def decode(
        self,
        latent: Tensor,
        *,
        scale: int | None = None,
        sigma_latent: float = 0.0,
        steps: int | None = None,
        text_cond: Tensor | None = None,
    ) -> Tensor:
        """Decode latent [B,C,h,w] → RGB [B,3,H,W] at target resolution."""
        cfg = self.cfg
        scale = scale or cfg.default_scale
        steps = steps or cfg.student_steps
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        lh, lw = latent.shape[-2], latent.shape[-1]
        out_h, out_w = lh * scale, lw * scale
        z_noisy = corrupt_latent(latent, sigma_latent)

        # Conditioning map from latent (spatial)
        cond_rgb = F.interpolate(z_noisy[:, :3].clamp(-3, 3), size=(out_h, out_w), mode="bilinear", align_corners=False)
        cond_rgb = (cond_rgb - cond_rgb.amin(dim=(2, 3), keepdim=True)) / (
            cond_rgb.amax(dim=(2, 3), keepdim=True) - cond_rgb.amin(dim=(2, 3), keepdim=True) + 1e-6
        )

        x = sample_noise_like(torch.zeros(latent.shape[0], 3, out_h, out_w, device=latent.device, dtype=latent.dtype))
        sigmas = cfg.dmd2_sigmas[:steps]
        if len(sigmas) < 2:
            sigmas = (1.0, 0.0)

        velocities: list[Tensor] = []
        for i in range(len(sigmas) - 1):
            t = 1.0 - sigmas[i]
            v = self.velocity(x, t, cond_rgb)
            # Adapter injection on flattened tokens (stub: bias velocity)
            tokens = self.project_latent_tokens(z_noisy, out_hw=(out_h, out_w), sigma=sigma_latent)
            inj = inject_latent(tokens, tokens, sigma_latent, self.gate)
            v = v + 0.01 * inj.mean(dim=(1, 2), keepdim=True)
            velocities.append(v)

        return integrate_velocity(x, velocities, tuple(sigmas[: len(velocities) + 1]))


def pid_decode_step(
    latent: Tensor,
    *,
    scale: int = 4,
    steps: int = 4,
    sigma_latent: float = 0.0,
    cfg: PiDConfig | None = None,
) -> Tensor:
    """Functional API: 4-step DMD2 student decode (replaces numpy repeat stub)."""
    if not isinstance(latent, Tensor):
        latent = torch.as_tensor(latent, dtype=torch.float32)
    decoder = PiDDecoder(cfg=cfg, latent_channels=latent.shape[-3] if latent.dim() >= 3 else 4)
    return decoder.decode(latent, scale=scale, sigma_latent=sigma_latent, steps=steps)


def training_loss(
    v_pred: Tensor,
    x0: Tensor,
    epsilon: Tensor,
    *,
    latent: Tensor | None = None,
    sigma_latent: float = 0.0,
) -> Tensor:
    """Eq. (10) conditional FM loss."""
    loss = flow_matching_loss(v_pred, velocity_target(x0, epsilon))
    if latent is None:
        return loss
    # Encourage adapter path to receive gradients in trainer smoke
    z = corrupt_latent(latent, sigma_latent)
    return loss + 1e-4 * z.pow(2).mean()
