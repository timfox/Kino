"""Dual-stream DiT stub: continuous audio + discrete text (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.uat.config import UATConfig


class DualStreamDiTStub(nn.Module):
    """Layer-wise audio↔text mutual conditioning (§3.2)."""

    def __init__(self, cfg: UATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UATConfig()
        d = min(self.cfg.hidden_dim, 256)  # stub width
        self.audio_in = nn.Linear(1, d)
        self.text_in = nn.Embedding(self.cfg.demo_vocab, d)
        self.blocks = nn.ModuleList(
            [DualStreamBlock(d) for _ in range(min(4, self.cfg.dit_blocks))]
        )
        self.audio_vel = nn.Linear(d, 1)
        self.text_logits = nn.Linear(d, self.cfg.demo_vocab)

    def forward(
        self,
        zt: Tensor,
        text_ids: Tensor,
        t_audio: Tensor,
        *,
        text_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # zt: (B, L), text_ids: (B, T)
        za = self.audio_in(zt.unsqueeze(-1))
        ht = self.text_in(text_ids)
        t_emb = t_audio.view(-1, 1, 1)
        za = za + t_emb
        for block in self.blocks:
            za, ht = block(za, ht)
        v_pred = self.audio_vel(za).squeeze(-1).reshape(zt.shape)
        logits = self.text_logits(ht)
        if text_mask is not None:
            logits = logits * text_mask.unsqueeze(-1)
        return v_pred, logits


class DualStreamBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.audio_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.ff_a = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ff_t = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, za: Tensor, ht: Tensor) -> tuple[Tensor, Tensor]:
        za2, _ = self.audio_attn(za, ht, ht)
        za = self.norm_a(za + za2)
        za = za + self.ff_a(za)
        ht2, _ = self.text_attn(ht, za, za)
        ht = self.norm_t(ht + ht2)
        ht = ht + self.ff_t(ht)
        return za, ht


def velocity_mse(v_pred: Tensor, v_target: Tensor) -> Tensor:
    return F.mse_loss(v_pred.reshape_as(v_target), v_target)


def masked_ce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy on masked token positions only."""
    b, t, v = logits.shape
    flat_l = logits.reshape(b * t, v)
    flat_y = targets.reshape(b * t)
    flat_m = mask.reshape(b * t).bool()
    if not flat_m.any():
        return logits.new_zeros(())
    return F.cross_entropy(flat_l[flat_m], flat_y[flat_m])
