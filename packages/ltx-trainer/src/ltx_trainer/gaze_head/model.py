"""Conditional VAE with GRU encoder/decoder for gaze-head coordination (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.gaze_head.config import GazeHeadConfig
from ltx_trainer.gaze_head.losses import cvae_loss


class GazeHeadCoordinationCVAE(nn.Module):
    """cVAE: gaze g_{1:T} -> head ĥ_{1:T} with optional context c_{1:2}."""

    def __init__(self, cfg: GazeHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GazeHeadConfig()
        self.cfg = cfg
        in_dim = 2 + 2 + cfg.context_frames * 2  # gaze + head + context poses
        self.enc_in = nn.Linear(in_dim, cfg.hidden_dim)
        self.encoder_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        dec_in = cfg.latent_dim + 2 + cfg.context_frames * 2
        self.dec_in = nn.Linear(dec_in, cfg.hidden_dim)
        self.decoder_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.head_out = nn.Linear(cfg.hidden_dim, 2)

    def _pack(
        self,
        gaze: Tensor,
        head: Tensor | None,
        context: Tensor | None,
    ) -> Tensor:
        """gaze (B,T,2); head (B,T,2) or None; context (B,C,2) -> (B,T,F)."""
        parts = [gaze]
        if head is None:
            parts.append(torch.zeros_like(gaze))
        else:
            parts.append(head)
        if context is None:
            b, t, _ = gaze.shape
            ctx = torch.zeros(b, self.cfg.context_frames, 2, device=gaze.device, dtype=gaze.dtype)
        else:
            ctx = context
        ctx_exp = ctx.unsqueeze(1).expand(-1, gaze.shape[1], -1, -1).reshape(gaze.shape[0], gaze.shape[1], -1)
        parts.append(ctx_exp)
        return torch.cat(parts, dim=-1)

    def encode(
        self,
        gaze: Tensor,
        head: Tensor,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = torch.relu(self.enc_in(self._pack(gaze, head, context)))
        _, h = self.encoder_gru(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(
        self,
        gaze: Tensor,
        z: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        b, t, _ = gaze.shape
        z_rep = z.unsqueeze(1).expand(-1, t, -1)
        if context is None:
            ctx = torch.zeros(b, self.cfg.context_frames, 2, device=gaze.device, dtype=gaze.dtype)
        else:
            ctx = context
        ctx_exp = ctx.unsqueeze(1).expand(-1, t, -1, -1).reshape(b, t, -1)
        dec_x = torch.relu(self.dec_in(torch.cat([z_rep, gaze, ctx_exp], dim=-1)))
        out, _ = self.decoder_gru(dec_x)
        return self.head_out(out)

    def forward(
        self,
        gaze: Tensor,
        head: Tensor,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(gaze, head, context)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        pred = self.decode(gaze, z, context)
        return pred, mu, logvar

    def sample_head(
        self,
        gaze: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Inference: z ~ N(0, I) (Sec. 2.2)."""
        b = gaze.shape[0]
        z = torch.randn(b, self.cfg.latent_dim, device=gaze.device, dtype=gaze.dtype)
        return self.decode(gaze, z, context)

    def training_loss(
        self,
        gaze: Tensor,
        head: Tensor,
        context: Tensor | None = None,
        *,
        kl_weight: float | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        pred, mu, logvar = self.forward(gaze, head, context)
        kw = self.cfg.kl_weight if kl_weight is None else kl_weight
        return cvae_loss(pred, head, mu, logvar, kl_weight=kw)
