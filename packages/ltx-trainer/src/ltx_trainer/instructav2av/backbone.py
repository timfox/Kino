"""Dual-stream editing backbone with source concat (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.instructav2av.config import InstructAV2AVConfig
from ltx_trainer.instructav2av.flow import dual_flow_loss, build_noisy_latents, velocity_target
from ltx_trainer.instructav2av.siga import SIGAModule


class _StreamVelocity(nn.Module):
    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels // 2, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DualStreamEditor(nn.Module):
    """
    Symmetric video/audio DiT stub: self-attn → SIGA → optional cross-modal attn.

    Source latents are channel-concatenated with noisy latents before denoising.
    """

    def __init__(
        self,
        *,
        cfg: InstructAV2AVConfig | None = None,
        latent_channels: int = 4,
        token_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = cfg or InstructAV2AVConfig()
        self.latent_channels = latent_channels
        self.token_dim = token_dim
        concat_ch = latent_channels * 2 if self.cfg.use_source_concat else latent_channels

        self.siga_v = SIGAModule(token_dim)
        self.siga_a = SIGAModule(token_dim)
        self.vel_v = _StreamVelocity(concat_ch)
        self.vel_a = _StreamVelocity(concat_ch)

        self.proj_v = nn.Conv2d(latent_channels, token_dim, 1)
        self.proj_a = nn.Conv2d(latent_channels, token_dim, 1)
        self.proj_inst = nn.Linear(token_dim, token_dim)

        self.cross_va = nn.Linear(token_dim, token_dim)
        self.cross_av = nn.Linear(token_dim, token_dim)

    def _concat_source(
        self,
        z_t: Tensor,
        z_s: Tensor,
    ) -> Tensor:
        if not self.cfg.use_source_concat:
            return z_t
        return torch.cat([z_t, z_s], dim=1)

    def _tokens_from_latent(self, z: Tensor, proj: nn.Module) -> Tensor:
        h = proj(z)
        return h.flatten(2).transpose(1, 2)

    def forward(
        self,
        zt_v: Tensor,
        zt_a: Tensor,
        zs_v: Tensor,
        zs_a: Tensor,
        inst_emb: Tensor,
        *,
        t: float,
        cross_modal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Predict velocity fields (û_v, û_a)."""
        xv = self._concat_source(zt_v, zs_v)
        xa = self._concat_source(zt_a, zs_a)

        fh_v = self._tokens_from_latent(zt_v, self.proj_v)
        fh_a = self._tokens_from_latent(zt_a, self.proj_a)
        fx_v = self._tokens_from_latent(zs_v, self.proj_v)
        fx_a = self._tokens_from_latent(zs_a, self.proj_a)

        if inst_emb.dim() == 2:
            fc = self.proj_inst(inst_emb).unsqueeze(1).expand(-1, fh_v.shape[1], -1)
        else:
            fc = self.proj_inst(inst_emb)

        if self.cfg.use_siga:
            fh_v, _ = self.siga_v(fh_v, fx_v, fc)
            fh_a, _ = self.siga_a(fh_a, fx_a, fc)

        if cross_modal:
            fh_v = fh_v + self.cross_va(fh_a.mean(dim=1, keepdim=True))
            fh_a = fh_a + self.cross_av(fh_v.mean(dim=1, keepdim=True))

        _ = (fh_v, fh_a)  # SIGA token path (full DiT blocks are external)
        return self.vel_v(xv), self.vel_a(xa)

    def training_step(
        self,
        z1_v: Tensor,
        z1_a: Tensor,
        zs_v: Tensor,
        zs_a: Tensor,
        inst_emb: Tensor,
        *,
        t: float | None = None,
        cross_modal: bool = True,
    ) -> dict[str, torch.Tensor]:
        t = float(t if t is not None else torch.rand(1).item())
        zt_v, zt_a, eps_v, eps_a = build_noisy_latents(z1_v, z1_a, t)
        u_v = velocity_target(z1_v, eps_v)
        u_a = velocity_target(z1_a, eps_a)
        u_hat_v, u_hat_a = self.forward(
            zt_v, zt_a, zs_v, zs_a, inst_emb, t=t, cross_modal=cross_modal
        )
        loss = dual_flow_loss(
            u_hat_v,
            u_hat_a,
            u_v,
            u_a,
            lambda_v=self.cfg.lambda_video,
            lambda_a=self.cfg.lambda_audio,
        )
        return {"loss": loss, "t": torch.tensor(t)}
