"""X2HDR frozen-VAE + LoRA flow stub (arXiv:2602.04814)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lumivid.lora_stub import LumiVidLoRAStub, flow_matching_loss
from ltx_trainer.lumivid.vae_stub import VaeStub
from ltx_trainer.x2hdr.pu21_codec import (
    scene_linear_to_vae_pixels,
    vae_denormalize,
    vae_normalize,
    vae_pixels_to_scene_linear,
)


@dataclass
class X2HdrConfig:
    latent_ch: int = 16
    denoise_steps: int = 11
    lora_rank: int = 8
    l_peak: float = 4000.0


class X2Hdr(nn.Module):
    """PU21-aligned HDR generation stub with frozen VAE surrogate + LoRA flow."""

    def __init__(self, cfg: X2HdrConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or X2HdrConfig()
        self.vae = VaeStub(latent_ch=self.cfg.latent_ch)
        self.lora = LumiVidLoRAStub(latent_ch=self.cfg.latent_ch, rank=self.cfg.lora_rank)

    def encode_sdr_reference(self, sdr_01: Tensor) -> Tensor:
        return self.vae.encode(vae_normalize(sdr_01))

    def encode_hdr_target(self, hdr_linear: Tensor) -> Tensor:
        pu21 = scene_linear_to_vae_pixels(hdr_linear, l_peak=self.cfg.l_peak)
        return self.vae.encode(vae_normalize(pu21))

    def decode_hdr_latent(self, z: Tensor) -> Tensor:
        disp = vae_denormalize(self.vae.decode(z))
        return vae_pixels_to_scene_linear(disp, l_peak=self.cfg.l_peak)

    @torch.no_grad()
    def infer(self, sdr_01: Tensor, *, steps: int | None = None) -> Tensor:
        self.eval()
        z_ref = self.encode_sdr_reference(sdr_01 if sdr_01.dim() == 4 else sdr_01.unsqueeze(0))
        n = steps or self.cfg.denoise_steps
        z = torch.randn_like(z_ref)
        for i in range(n):
            t = torch.full((z.shape[0],), (i + 1) / n, device=z.device, dtype=z.dtype)
            z = z + self.lora(z, z_ref, t) / n
        out = self.decode_hdr_latent(z)
        return out.squeeze(0) if sdr_01.dim() == 3 else out

    def training_step(self, sdr_01: Tensor, hdr_linear: Tensor) -> tuple[Tensor, dict[str, float]]:
        sdr = sdr_01.clamp(0, 1) if sdr_01.max() <= 1.0 else tone_map(sdr_01)
        z_ref = self.encode_sdr_reference(sdr.unsqueeze(0) if sdr.dim() == 3 else sdr)
        z_tgt = self.encode_hdr_target(hdr_linear.unsqueeze(0) if hdr_linear.dim() == 3 else hdr_linear)
        loss = flow_matching_loss(self.lora, z_tgt, z_ref)
        with torch.no_grad():
            pred = self.decode_hdr_latent(z_tgt)
            if pred.dim() == 4:
                pred = pred.squeeze(0)
        recon = torch.nn.functional.l1_loss(
            scene_linear_to_vae_pixels(hdr_linear, l_peak=self.cfg.l_peak),
            scene_linear_to_vae_pixels(pred.clamp(min=0.0), l_peak=self.cfg.l_peak),
        )
        total = loss + 0.1 * recon
        return total, {"loss_flow": float(loss.detach()), "loss_recon_pu21": float(recon.detach())}


def tone_map(x: Tensor) -> Tensor:
    from ltx_trainer.lumivid.degrade import tone_map_reinhard

    return tone_map_reinhard(x)
