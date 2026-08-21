"""Slot attention aggregator + decoder stubs — Sec. 3.1."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SlotAttention(nn.Module):
    """Minimal slot attention (Locatello et al., 2020)."""

    def __init__(self, num_slots: int, slot_dim: int, feature_dim: int, iters: int = 3) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.decoder_mlp = nn.Sequential(
            nn.Linear(slot_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: Tensor, queries: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """features [B, N, D], queries [B, S, D] optional."""
        b, n, _ = features.shape
        if queries is None:
            mu = self.slots_mu.expand(b, -1, -1)
            sigma = self.slots_log_sigma.exp().expand(b, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = queries
        k = self.project_k(self.norm_input(features))
        v = self.project_v(self.norm_input(features))
        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            logits = torch.einsum("bsd,bnd->bsn", q, k) / (q.shape[-1] ** 0.5)
            attn = torch.softmax(logits, dim=-1)
            updates = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]), slots_prev.reshape(-1, slots.shape[-1]))
            slots = slots.reshape(b, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_pre(slots))
        return slots, attn

    def decode(self, slots: Tensor) -> Tensor:
        """Slot-mixture reconstruction [B, S, D] -> [B, D] mean pool decode."""
        return self.decoder_mlp(slots).mean(dim=1)


class TransitionModule(nn.Module):
    """RandSF.Q-style transition with stochasticity ε — Eq. (1b)."""

    def __init__(self, slot_dim: int, feature_dim: int, *, backward: bool = False) -> None:
        super().__init__()
        self.backward = backward
        self.gru = nn.GRUCell(slot_dim + feature_dim, slot_dim)
        self.time_embed = nn.Parameter(torch.randn(1, slot_dim) * (1.0 if backward else -1.0))

    def forward(self, prev_slots: Tensor, frame_feat: Tensor, *, noise_std: float = 0.0) -> Tensor:
        b, s, d = prev_slots.shape
        feat = frame_feat.mean(dim=1).unsqueeze(1).expand(-1, s, -1)
        combined = torch.cat([prev_slots, feat], dim=-1)
        flat = combined.reshape(b * s, -1)
        out = self.gru(flat, prev_slots.reshape(b * s, d))
        out = out.reshape(b, s, d) + self.time_embed
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out
