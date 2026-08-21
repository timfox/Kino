"""DiffHDR video diffusion model (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.diffhdr.cfa import ContextFocusedAttention
from ltx_trainer.diffhdr.config import DiffHDRConfig
from ltx_trainer.diffhdr.log_gamma import inverse_log_gamma_map, log_gamma_map
from ltx_trainer.diffhdr.lora import LoRALinear
from ltx_trainer.diffhdr.losses import flow_matching_loss, sample_flow_pair
from ltx_trainer.diffhdr.mask import detect_exposure_masks, ema_smooth_masks
from ltx_trainer.diffhdr.vae_stub import VideoVAEStub


@dataclass
class DiffHDROutput:
    hdr: Tensor
    latent: Tensor


class DiTBlockStub(nn.Module):
    def __init__(self, dim: int, rank: int = 32) -> None:
        super().__init__()
        self.ff = LoRALinear(nn.Linear(dim, dim * 4), rank)
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim * 4, dim)
        for p in self.out.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return x + self.out(torch.relu(self.ff(self.ln(x))))


class DiffHDR(nn.Module):
    """LDR-to-HDR video diffusion with Log-Gamma VAE compatibility."""

    def __init__(self, cfg: DiffHDRConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DiffHDRConfig()
        d = self.cfg.hidden_dim
        ld = self.cfg.latent_dim
        self.vae = VideoVAEStub(ld)
        self.dit = nn.ModuleList([DiTBlockStub(d, self.cfg.lora_rank) for _ in range(3)])
        self.cfa = ContextFocusedAttention(d)
        self.token_in = nn.Linear(ld, d)
        self.velocity_head = nn.Linear(d, ld)

    def _to_vae_input(self, radiance: Tensor) -> Tensor:
        if radiance.dim() == 4:
            radiance = radiance.unsqueeze(0)
        b, t, c, h, w = radiance.shape
        flat = radiance.reshape(b * t, c, h, w)
        mapped = log_gamma_map(flat, m=self.cfg.log_gamma_m, gamma=self.cfg.log_gamma_gamma)
        return mapped.reshape(b, t, c, h, w)

    def encode(self, radiance: Tensor) -> Tensor:
        return self.vae.encode(self._to_vae_input(radiance))

    def decode_hdr(self, z: Tensor) -> Tensor:
        decoded = self.vae.decode(z)
        b, t, c, h, w = decoded.shape
        flat = decoded.reshape(b * t, c, h, w)
        hdr = inverse_log_gamma_map(flat, m=self.cfg.log_gamma_m, gamma=self.cfg.log_gamma_gamma)
        return hdr.reshape(b, t, c, h, w)

    def _process_tokens(self, z: Tensor, ldr: Tensor, *, use_cfa: bool) -> Tensor:
        b, c, t, h, w = z.shape
        tokens = self.token_in(z.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c))
        if self.cfg.use_mask and use_cfa:
            over_list, under_list = [], []
            for i in range(ldr.shape[1]):
                mo, mu = detect_exposure_masks(
                    ldr[0, i], tau_high=self.cfg.tau_high, tau_low=self.cfg.tau_low
                )
                over_list.append(mo)
                under_list.append(mu)
            m_over = ema_smooth_masks(over_list, alpha=self.cfg.mask_ema_alpha)[-1].unsqueeze(0)
            m_under = ema_smooth_masks(under_list, alpha=self.cfg.mask_ema_alpha)[-1].unsqueeze(0)
            tokens = self.cfa(tokens, m_over, m_under)
        for block in self.dit:
            tokens = block(tokens)
        return tokens

    def predict_velocity(self, ldr: Tensor) -> Tensor:
        if ldr.dim() == 4:
            ldr = ldr.unsqueeze(0)
        z = self.encode(ldr)
        b, c, t, h, w = z.shape
        tokens = self._process_tokens(z, ldr, use_cfa=self.cfg.use_cfa)
        return self.velocity_head(tokens.mean(dim=1)).reshape(b, c, 1, 1, 1).expand_as(z)

    def forward(self, ldr: Tensor, *, hdr_target: Tensor | None = None) -> DiffHDROutput:
        if ldr.dim() == 4:
            ldr = ldr.unsqueeze(0)
        z_ctx = self.encode(ldr)
        pred_v = self.predict_velocity(ldr)
        if hdr_target is not None:
            z1 = self.encode(hdr_target)
            _xt, _t, target_v = sample_flow_pair(z1)
            self._last_flow_loss = flow_matching_loss(pred_v, target_v)
        else:
            self._last_flow_loss = None
        pred_z = z_ctx + pred_v
        hdr = self.decode_hdr(pred_z)
        return DiffHDROutput(hdr=hdr, latent=pred_z)
