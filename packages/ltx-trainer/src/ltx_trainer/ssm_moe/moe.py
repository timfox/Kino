"""Top-k mixture-of-experts feed-forward and load-balancing losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """SwiGLU-style expert MLP."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoERouter(nn.Module):
    """Token router producing top-k expert indices and weights."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out_weights: (B, L, K) normalized gate weights for selected experts
            indices: (B, L, K) expert indices
            logits: (B, L, E) full router logits (for aux loss)
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_w, top_i, logits


class MoEFeedForward(nn.Module):
    """Sparse top-k MoE FFN (Switch / Mixtral style)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.router = MoERouter(d_model, num_experts, top_k=top_k)
        self.experts = nn.ModuleList(ExpertFFN(d_model, d_ff) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, d_model = x.shape
        weights, indices, logits = self.router(x)
        flat_x = x.reshape(-1, d_model)
        flat_out = torch.zeros_like(flat_x)
        flat_idx = indices.reshape(-1, indices.shape[-1])
        flat_w = weights.reshape(-1, weights.shape[-1])

        for slot in range(indices.shape[-1]):
            for e in range(len(self.experts)):
                mask = flat_idx[:, slot] == e
                if not mask.any():
                    continue
                expert_out = self.experts[e](flat_x[mask])
                flat_out[mask] = flat_out[mask] + flat_w[mask, slot : slot + 1] * expert_out

        return flat_out.reshape(bsz, seqlen, d_model), logits


def load_balance_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Switch Transformer auxiliary load-balancing loss.

    Encourages uniform expert utilization across tokens.
    """
    probs = F.softmax(router_logits, dim=-1)
    top1 = probs.argmax(dim=-1)
    one_hot = F.one_hot(top1, num_classes=num_experts).float()
    freq = one_hot.mean(dim=(0, 1))
    mean_prob = probs.mean(dim=(0, 1))
    return num_experts * (freq * mean_prob).sum()
