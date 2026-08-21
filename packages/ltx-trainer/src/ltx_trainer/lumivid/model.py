"""LumiVid SDR→HDR video stub pipeline (arXiv:2604.11788)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lumivid.degrade import apply_reference_degradations, tone_map_reinhard
from ltx_trainer.lumivid.logc3_codec import (
    scene_linear_to_vae_pixels,
    vae_denormalize,
    vae_normalize,
    vae_pixels_to_scene_linear,
)
from ltx_trainer.lumivid.lora_stub import LumiVidLoRAStub, flow_matching_loss
from ltx_trainer.lumivid.vae_stub import VaeStub


@dataclass
class LumiVidConfig:
    latent_ch: int = 16
    denoise_steps: int = 11
    lora_rank: int = 8


class LumiVid(nn.Module):
    """LogC3-aligned HDR video generation stub with frozen VAE surrogate + LoRA flow."""

    def __init__(self, cfg: LumiVidConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LumiVidConfig()
        self.vae = VaeStub(latent_ch=self.cfg.latent_ch)
        self.lora = LumiVidLoRAStub(latent_ch=self.cfg.latent_ch, rank=self.cfg.lora_rank)

    def encode_sdr_reference(self, sdr_01: Tensor) -> Tensor:
        return self.vae.encode(vae_normalize(sdr_01))

    def encode_hdr_target(self, hdr_linear: Tensor) -> Tensor:
        logc3 = scene_linear_to_vae_pixels(hdr_linear)
        return self.vae.encode(vae_normalize(logc3))

    def decode_hdr_latent(self, z: Tensor) -> Tensor:
        disp = vae_denormalize(self.vae.decode(z))
        return vae_pixels_to_scene_linear(disp)

    @torch.no_grad()
    def infer(self, sdr_01: Tensor, *, steps: int | None = None) -> Tensor:
        """SDR [0,1] → scene-linear HDR (single frame or batch)."""
        self.eval()
        sdr_deg = apply_reference_degradations(sdr_01)
        z_ref = self.encode_sdr_reference(sdr_deg)
        n = steps or self.cfg.denoise_steps
        z = torch.randn_like(z_ref)
        for i in range(n):
            t = torch.full((z.shape[0],), (i + 1) / n, device=z.device, dtype=z.dtype)
            v = self.lora(z, z_ref, t)
            z = z + v / n
        return self.decode_hdr_latent(z)

    def training_step(self, sdr_01: Tensor, hdr_linear: Tensor) -> tuple[Tensor, dict[str, float]]:
        sdr_tm = tone_map_reinhard(sdr_01) if hdr_linear.max() > 1.5 else sdr_01.clamp(0, 1)
        sdr_deg = apply_reference_degradations(sdr_tm)
        z_ref = self.encode_sdr_reference(sdr_deg)
        z_tgt = self.encode_hdr_target(hdr_linear)
        loss = flow_matching_loss(self.lora, z_tgt, z_ref)
        with torch.no_grad():
            pred = self.decode_hdr_latent(z_tgt)
            if pred.dim() == 4 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
        recon = torch.nn.functional.l1_loss(
            scene_linear_to_vae_pixels(hdr_linear),
            scene_linear_to_vae_pixels(pred.clamp(min=0.0)),
        )
        total = loss + 0.1 * recon
        return total, {
            "loss_flow": float(loss.detach()),
            "loss_recon_logc3": float(recon.detach()),
            "loss_total": float(total.detach()),
        }
