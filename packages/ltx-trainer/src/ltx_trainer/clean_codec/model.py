"""Disentangled VQ-VAE-style mel codec stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.clean_codec.config import CleanCodecConfig
from ltx_trainer.clean_codec.quantize import fsq_quantize


class CleanCodecStub(nn.Module):
    """Local encoder → FSQ → global emb → dual decoder (acoustic + semantic)."""

    def __init__(self, cfg: CleanCodecConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CleanCodecConfig()
        d = 128
        in_dim = self.cfg.demo_mel_bins
        self.local_enc = nn.Sequential(
            nn.Conv1d(in_dim, d, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=7, padding=3, stride=5),
        )
        self.global_enc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d, cfg.global_emb_dim),
        )
        self.acoustic_dec = nn.ConvTranspose1d(d, in_dim, kernel_size=7, stride=5, padding=3, output_padding=4)
        self.semantic_dec = nn.Linear(d, d)
        self.ssl_proj = nn.Linear(d, d)

    def encode(self, mel: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # mel: (B, F, T) → channels first
        x = mel if mel.dim() == 3 and mel.shape[1] == self.cfg.demo_mel_bins else mel.transpose(1, 2)
        h = self.local_enc(x)
        z = h.mean(dim=-1)
        z_q = fsq_quantize(z, self.cfg.fsq_levels)
        g = self.global_enc(h)
        return z_q, g, h

    def decode(self, z_q: Tensor, g: Tensor, length: int) -> tuple[Tensor, Tensor]:
        b = z_q.shape[0]
        d = 128
        h = z_q.unsqueeze(-1).expand(b, d, length)
        m_hat = self.acoustic_dec(h)
        s_hat = self.ssl_proj(z_q)
        return m_hat, s_hat

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        z_q, g, h = self.encode(mel)
        t_len = h.shape[-1]
        m_hat, s_hat = self.decode(z_q, g, t_len)
        return {"m_hat": m_hat, "s_hat": s_hat, "g_hat": g, "z_q": z_q}
