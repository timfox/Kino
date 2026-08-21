"""LumaFlux end-to-end model (Fig. 2–3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lumaflux.backbone import MMDiTBlockStub
from ltx_trainer.lumaflux.color import pq_oetf, rgb_to_yuv2020, yuv_to_rgb2020
from ltx_trainer.lumaflux.config import LumaFluxConfig
from ltx_trainer.lumaflux.coupler import HDRResidualCoupler
from ltx_trainer.lumaflux.modulation import TimestepLayerModulator
from ltx_trainer.lumaflux.pcm import PCMFiLM, PerceptualConnector, SigLIPStub
from ltx_trainer.lumaflux.pga import PGALoRA
from ltx_trainer.lumaflux.physical import PhysicalFeatureExtractor
from ltx_trainer.lumaflux.rqs import RQSToneField


@dataclass
class LumaFluxOutput:
    hdr: Tensor
    latent: Tensor
    spline_logits: Tensor


class VAEStub(nn.Module):
    """Frozen VAE encode/decode stub."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, latent_dim, 3, stride=2, padding=1),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


class LumaFlux(nn.Module):
    """Physically and perceptually guided DiT adapter for SDR→HDR ITM."""

    def __init__(self, cfg: LumaFluxConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LumaFluxConfig()
        d = self.cfg.hidden_dim
        self.physical = PhysicalFeatureExtractor(d)
        self.siglip = SigLIPStub(self.cfg.embed_dim)
        self.connector = PerceptualConnector(self.cfg.embed_dim, d)
        self.modulator = TimestepLayerModulator(d, self.cfg.num_blocks)
        self.blocks = nn.ModuleList([MMDiTBlockStub(d) for _ in range(self.cfg.num_blocks)])
        self.pga_layers = nn.ModuleList([PGALoRA(d, self.cfg.lora_rank, d) for _ in range(self.cfg.num_blocks)])
        self.pcm_layers = nn.ModuleList([PCMFiLM(d, d) for _ in range(self.cfg.num_blocks)])
        self.couplers = nn.ModuleList([HDRResidualCoupler(d, d, d) for _ in range(self.cfg.num_blocks)])
        self.vae = VAEStub(d)
        self.rqs = RQSToneField(self.cfg.rqs_knots, d)
        self.token_proj = nn.Conv2d(d, d, 1)

    def _tokens_from_latent(self, z: Tensor) -> Tensor:
        t = self.token_proj(z)
        b, c, h, w = t.shape
        return t.flatten(2).transpose(1, 2)

    def _latent_from_tokens(self, tokens: Tensor, h: int, w: int) -> Tensor:
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, sdr: Tensor, *, t: float = 0.5) -> LumaFluxOutput:
        if sdr.dim() == 3:
            sdr = sdr.unsqueeze(0)
        t_phys, g_global, spectral = self.physical(sdr)
        t_perc = self.connector(self.siglip(sdr))
        z0, x_vae = self.vae(sdr)
        _, _, h, w = z0.shape
        z = self._tokens_from_latent(z0)
        t_tensor = torch.tensor([t], device=sdr.device, dtype=sdr.dtype)

        for i, block in enumerate(self.blocks):
            mod = self.modulator(t_tensor, i)
            v = z
            if self.cfg.use_pga:
                v = self.pga_layers[i](z, t_phys, g_global, spectral, mod)
            h_pcm = None
            if self.cfg.use_pcm:
                h_pcm = self.pcm_layers[i](z, t_perc, mod)
            z_res = block(z, v_override=v, h_override=h_pcm)
            if self.cfg.use_coupler:
                z_res = self.couplers[i](z_res, t_phys, t_perc, mod)
            z = z_res

        z_out = self._latent_from_tokens(z, h, w)
        with torch.no_grad():
            x_out = self.vae.decode(z_out)
        y, u, v = rgb_to_yuv2020(x_out)
        logits = self.rqs.param_head(y)
        y_hat, u_hat, v_hat = self.rqs(y, u, v)
        hdr = pq_oetf(yuv_to_rgb2020(y_hat, u_hat, v_hat))
        return LumaFluxOutput(hdr=hdr, latent=z_out, spline_logits=logits)
