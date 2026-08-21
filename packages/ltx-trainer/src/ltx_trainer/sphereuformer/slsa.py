"""Spherical Local Self-Attention (SLSA, Fig. 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.sphereuformer.icosphere import k_order_neighbors
from ltx_trainer.sphereuformer.positional import RelativePositionBias, VerticalGlobalPE


class SphericalLocalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        c_head: float,
        c_win: int,
        n_nodes: int,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.c_win = c_win
        head_dim = max(1, int((dim / num_heads) * c_head))
        self.head_dim = head_dim
        inner = head_dim * num_heads
        self.qkv = nn.Linear(dim, inner * 3)
        self.out_proj = nn.Linear(inner, dim)
        self.rel_bias = RelativePositionBias(7, num_heads) if use_rel_pos else None
        self.use_abs_pos = use_abs_pos
        nbr = k_order_neighbors(min(n_nodes, 512), c_win=c_win)
        self.register_buffer("_nbr", nbr, persistent=False)

    def forward(self, x: Tensor, phi: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        k_n = self._nbr.shape[0]
        idx = self._nbr[:, :n] % n
        k_stack = torch.stack([k[:, idx[j]] for j in range(k_n)], dim=2)
        v_stack = torch.stack([v[:, idx[j]] for j in range(k_n)], dim=2)
        scores = torch.einsum("bnhd,bnkhd->bnhk", q, k_stack) / (self.head_dim**0.5)
        if self.rel_bias is not None:
            dt = torch.zeros(b, n, k_n, device=x.device)
            dp = torch.linspace(-1, 1, k_n, device=x.device).view(1, 1, -1).expand(b, n, -1)
            bias = self.rel_bias(dt, dp).unsqueeze(2)
            scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bnhk,bnkhd->bnhd", attn, v_stack)
        return self.out_proj(out.reshape(b, n, -1))


class SphericalAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, c_head: float, c_win: int, n_nodes: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.vert_pe = VerticalGlobalPE(dim)
        self.slsa = SphericalLocalSelfAttention(
            dim, num_heads=num_heads, c_head=c_head, c_win=c_win, n_nodes=n_nodes
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor, phi: Tensor) -> Tensor:
        h = self.norm1(x)
        h = h + self.vert_pe(phi[: h.shape[1]]).unsqueeze(0)
        x = x + self.slsa(h, phi)
        return x + self.mlp(self.norm2(x))
