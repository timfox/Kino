"""Memory Queries: gated eviction update (Echo-Infinity §3.3, Eq. 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from ltx_trainer.echo_infinity.config import EchoInfinityConfig


def memory_token_count(cfg: EchoInfinityConfig) -> int:
    return cfg.num_memory_query_frames * cfg.tokens_per_frame


class MemoryQueryProjector(nn.Module):
    """Shared K/V projections W^Q_k, W^Q_v (layer-shared in paper)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        return self.w_k(queries), self.w_v(queries)


class GatedMemoryResidual(nn.Module):
    """g = σ([Q; Q̃] W_gate),  Q ← g ⊙ Q + (1-g) ⊙ Q̃."""

    def __init__(self, dim: int, gate_bias: float = 2.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, gate_bias)

    def forward(self, q: Tensor, q_tilde: Tensor) -> Tensor:
        gate_in = torch.cat([q, q_tilde], dim=-1)
        g = torch.sigmoid(self.w_gate(gate_in))
        return g * q + (1.0 - g) * q_tilde


class CrossAttentionMemoryEncoder(nn.Module):
    """Lenc cross-attention layers: Q attends to evicted K/V (§3.3)."""

    def __init__(self, dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, queries: Tensor, k_evict: Tensor, v_evict: Tensor) -> Tensor:
        """queries: (B, Tq, D); k_evict/v_evict: (B, Te, D)."""
        h = queries
        for layer in self.layers:
            attn_out, _ = layer(h, k_evict, v_evict, need_weights=False)
            h = self.out_norm(h + attn_out)
        return h


class MemoryQueryStack(nn.Module):
    """Full memory module: encoder + gate + optional K/V projection."""

    def __init__(self, cfg: EchoInfinityConfig) -> None:
        super().__init__()
        from ltx_trainer.echo_infinity.config import EchoInfinityConfig

        self.cfg = cfg
        dim = cfg.hidden_dim
        n_tokens = memory_token_count(cfg)
        self.queries = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.encoder = CrossAttentionMemoryEncoder(dim, cfg.memory_encoder_layers, cfg.num_heads)
        self.gate = GatedMemoryResidual(dim, gate_bias=cfg.gate_bias)
        self.projector = MemoryQueryProjector(dim)

    def init_queries(self, batch: int) -> Tensor:
        return self.queries.expand(batch, -1, -1)

    def update(
        self,
        q: Tensor,
        k_evict: Tensor,
        v_evict: Tensor,
        *,
        enabled: bool = True,
    ) -> Tensor:
        if not enabled:
            return q
        q_tilde = self.encoder(q, k_evict, v_evict)
        return self.gate(q, q_tilde)

    def project_kv(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return self.projector(q)


def update_memory_queries(
    q: Tensor,
    k_evict: Tensor,
    v_evict: Tensor,
    encoder: CrossAttentionMemoryEncoder,
    gate: GatedMemoryResidual,
) -> Tensor:
    """Functional Eq. (4) for tests without full stack."""
    q_tilde = encoder(q, k_evict, v_evict)
    return gate(q, q_tilde)
