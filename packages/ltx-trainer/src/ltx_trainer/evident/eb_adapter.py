"""Entity Bottleneck Adapter with slot attention (Sec. 4.2, Eq. 2–4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.evident.config import EVIDENTConfig


class SlotAttentionBlock(nn.Module):
    """Competitive slot assignment with GRU updates per frame."""

    def __init__(self, cfg: EVIDENTConfig) -> None:
        super().__init__()
        d = cfg.bottleneck_dim
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.num_slots = cfg.num_slots

    def forward(self, visual: Tensor, slots: Tensor) -> tuple[Tensor, Tensor]:
        """``visual`` (B, N, d), ``slots`` (B, K, d) → updated slots and assignment ˆA (B, N, K)."""
        b, n, d = visual.shape
        k = self.num_slots
        s = slots
        attn_hat = None
        for _ in range(3):
            q = self.wq(s)
            key = self.wk(visual)
            val = self.wv(visual)
            logits = torch.matmul(key, q.transpose(1, 2)) / (d**0.5)
            a = F.softmax(logits, dim=-1)
            a_sum = a.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attn_hat = a / a_sum
            z = torch.matmul(attn_hat.transpose(1, 2), val)
            s = self.gru(z.reshape(-1, d), s.reshape(-1, d)).reshape(b, k, d)
        assert attn_hat is not None
        return s, attn_hat


class EntityBottleneckAdapter(nn.Module):
    """Down-project → slot attention → token reconstruction → residual up-project."""

    def __init__(self, cfg: EVIDENTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_in, d = cfg.hidden_dim, cfg.bottleneck_dim
        self.down = nn.Linear(d_in, d)
        self.up = nn.Linear(d, d_in)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.slot_block = SlotAttentionBlock(cfg)
        self.slots_init = nn.Parameter(torch.randn(cfg.num_slots, d) * 0.02)

    def forward(
        self,
        x: Tensor,
        *,
        text_len: int,
        gating: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``x`` (B, L, D); last ``text_len`` tokens are text/query. Returns (x_out, attn_hat)."""
        b, seq_len, _ = x.shape
        t = max((seq_len - text_len) // (self.cfg.tokens_per_frame + 1), 1)
        n = self.cfg.tokens_per_frame
        vis_len = t * n
        x_down = self.down(x)
        vis = x_down[:, :vis_len].reshape(b, t, n, -1)
        slots = self.slots_init.unsqueeze(0).expand(b, -1, -1)
        recon_frames: list[Tensor] = []
        last_attn: Tensor | None = None
        for fi in range(t):
            s_out, attn = self.slot_block(vis[:, fi], slots)
            slots = s_out
            recon = torch.matmul(attn, s_out)
            if gating is not None:
                recon = recon * gating[:, fi].view(b, 1, 1)
            recon_frames.append(recon)
            last_attn = attn
        recon_vis = torch.cat(recon_frames, dim=1).reshape(b, vis_len, -1)
        txt = x_down[:, vis_len:]
        merged = torch.cat([recon_vis, txt], dim=1)
        delta = self.up(merged)
        return x + delta, last_attn if last_attn is not None else attn
