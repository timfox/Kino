"""Stub DA3 encoder, LoRA, latent diffusion, and RGB decoder (Sec. III-C)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.deblur_nvs.config import DeblurNVSConfig


class LoRALinear(nn.Module):
    """Low-rank adapter on a linear layer (Hu et al., ICLR 2022)."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x))


class DA3EncoderStub(nn.Module):
    """Student DA3 encoder with optional LoRA (Eq. 7)."""

    def __init__(self, cfg: DeblurNVSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, cfg.latent_channels, 3, padding=1),
        )
        self.lora = LoRALinear(nn.Linear(cfg.latent_channels, cfg.latent_channels), cfg.lora_rank)

    def forward(self, images: Tensor) -> Tensor:
        """``images``: ``(K, 3, H, W)`` → ``(K, C, H, W)`` latents."""
        z = self.conv(images)
        k, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        adapted = self.lora(flat).reshape(k, h, w, c).permute(0, 3, 1, 2)
        return adapted


class LatentDiffusionStub(nn.Module):
    """Context or target latent denoiser (Eq. 9, 13)."""

    def __init__(self, cfg: DeblurNVSConfig, *, use_camera: bool) -> None:
        super().__init__()
        self.use_camera = use_camera
        cam_in = 16 if use_camera else 0
        in_ch = cfg.latent_channels * 2 + cam_in
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, cfg.latent_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 3, padding=1),
        )

    def forward(
        self,
        z_noisy: Tensor,
        context: Tensor,
        *,
        camera: Tensor | None = None,
    ) -> Tensor:
        if self.use_camera:
            if camera is None:
                raise ValueError("target diffusion requires camera conditioning")
            cam = camera.view(1, -1, 1, 1).expand(z_noisy.shape[0], -1, z_noisy.shape[2], z_noisy.shape[3])
            x = torch.cat([z_noisy, context, cam], dim=1)
        else:
            x = torch.cat([z_noisy, context], dim=1)
        return self.net(x)


class RGBDecoderStub(nn.Module):
    """Lightweight ViT-style decoder (Sec. III-C, Eq. 14)."""

    def __init__(self, cfg: DeblurNVSConfig) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(cfg.latent_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: Tensor) -> Tensor:
        return self.head(latents)


def euler_denoise(
    model: LatentDiffusionStub,
    z_init: Tensor,
    context: Tensor,
    *,
    steps: int,
    camera: Tensor | None = None,
) -> Tensor:
    """Simple Euler integration for latent denoising smoke."""
    z = z_init.clone()
    dt = 1.0 / max(steps, 1)
    for _ in range(steps):
        eps = model(z, context, camera=camera)
        z = z - dt * eps
    return z


class DeblurNVSStub(nn.Module):
    """Full two-stage DeblurNVS stub."""

    def __init__(self, cfg: DeblurNVSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = DA3EncoderStub(cfg)
        self.ctx_diffusion = LatentDiffusionStub(cfg, use_camera=False)
        self.tgt_diffusion = LatentDiffusionStub(cfg, use_camera=True)
        self.decoder = RGBDecoderStub(cfg)

    def restore_context(self, blur_latents: Tensor) -> Tensor:
        ctx = blur_latents.mean(dim=0, keepdim=True).expand_as(blur_latents)
        return euler_denoise(
            self.ctx_diffusion,
            blur_latents,
            ctx,
            steps=self.cfg.context_steps,
        )

    def synthesize_target(
        self,
        restored_ctx: Tensor,
        *,
        camera: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        ctx = restored_ctx.mean(dim=0, keepdim=True)
        z0 = noise if noise is not None else torch.randn_like(ctx)
        return euler_denoise(
            self.tgt_diffusion,
            z0,
            ctx,
            steps=self.cfg.target_steps,
            camera=camera,
        )

    def decode_rgb(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)
