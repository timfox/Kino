"""Panoramic Sparse Attention (Sec. III-C, Eq. 7–11)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PanoramicSparseAttention(nn.Module):
    """Top-K sparse attention with position-aware gating (PSA)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        top_k: int = 32,
        gate_dim: int = 32,
        index_dim: int = 16,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.wq_sparse = nn.Linear(dim, dim)
        self.wk_sparse = nn.Linear(dim, dim)
        self.wv_sparse = nn.Linear(dim, dim)
        self.wq_g = nn.Linear(dim, gate_dim)
        self.wk_g = nn.Linear(dim, gate_dim)
        self.wq_i = nn.Linear(dim, index_dim)
        self.wk_i = nn.Linear(dim, index_dim)
        self.pe = nn.Parameter(torch.zeros(1, 1, 1))
        self.out_proj = nn.Linear(dim, dim)

    def _selector_scores(self, h: Tensor) -> Tensor:
        """I[t,s] relevance matrix (Eq. 7–8, simplified)."""
        b, l, d = h.shape
        scores = torch.zeros(b, l, l, device=h.device, dtype=h.dtype)
        for j in range(self.num_heads):
            qg = self.wq_g(h)
            kg = self.wk_g(h)
            qi = self.wq_i(h)
            ki = self.wk_i(h)
            gate = torch.sigmoid(
                (qg.unsqueeze(2) * kg.unsqueeze(1)).sum(dim=-1, keepdim=True) + self.pe
            ).squeeze(-1)
            relu_sim = F.relu(torch.bmm(qi, ki.transpose(1, 2)))
            scores = scores + gate * relu_sim
        return scores / max(self.num_heads, 1)

    def forward(self, h: Tensor) -> Tensor:
        b, l, d = h.shape
        q = self.wq_sparse(h)
        k = self.wk_sparse(h)
        v = self.wv_sparse(h)
        idx_scores = self._selector_scores(h)
        k_use = min(self.top_k, l)
        top_idx = torch.topk(idx_scores, k=k_use, dim=-1).indices
        out = torch.zeros_like(q)
        scale = d**-0.5
        for bi in range(b):
            for ti in range(l):
                sel = top_idx[bi, ti]
                q_t = q[bi, ti : ti + 1]
                k_t = k[bi, sel]
                v_t = v[bi, sel]
                attn = F.softmax((q_t @ k_t.transpose(0, 1)) * scale, dim=-1)
                out[bi, ti] = (attn @ v_t).squeeze(0)
        return self.out_proj(out)


class SimplifiedSparseAttention(nn.Module):
    """SSA without position gate (Supp. Eq. 14–17)."""

    def __init__(self, dim: int, num_heads: int = 4, top_k: int = 32, index_dim: int = 16) -> None:
        super().__init__()
        self.psa = PanoramicSparseAttention(dim, num_heads, top_k, gate_dim=index_dim, index_dim=index_dim)
        self.head_weights = nn.Linear(dim, num_heads)

    def forward(self, h: Tensor) -> Tensor:
        # SSA uses query-only head weights; stub delegates to PSA path for smoke
        _ = self.head_weights(h)
        return self.psa(h)
