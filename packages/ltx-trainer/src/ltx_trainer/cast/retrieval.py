"""Causal successor retrieval from memory (Sec. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig
from ltx_trainer.cast.simplex import normalize_simplex


class CausalRetrieval(nn.Module):
    """Multi-head causal retrieval of empirical successors r_t from past (h_s, p_{s+1})."""

    def __init__(self, cfg: CASTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "wq": nn.Linear(cfg.hidden_dim, cfg.retrieval_dim, bias=False),
                        "wk": nn.Linear(cfg.hidden_dim, cfg.retrieval_dim, bias=False),
                    }
                )
                for _ in range(cfg.retrieval_heads)
            ]
        )
        self.head_mix = nn.Linear(cfg.hidden_dim, cfg.retrieval_heads)

    def forward(self, hidden: Tensor, history_hidden: Tensor, history_next: Tensor) -> Tensor:
        """
        hidden: (B, H) current state encoding
        history_hidden: (B, T, H) causal past encodings (s < t)
        history_next: (B, T, D) empirical successors p_{s+1}
        """
        if history_hidden.shape[1] == 0:
            # no memory: persistence fallback handled by caller
            return history_next.new_zeros(history_next.shape[0], history_next.shape[-1])

        head_outputs: list[Tensor] = []
        scale = self.cfg.retrieval_dim ** -0.5
        for head in self.heads:
            q = head["wq"](hidden).unsqueeze(1)  # (B, 1, dr)
            k = head["wk"](history_hidden)  # (B, T, dr)
            scores = (q * k).sum(-1) * scale  # (B, T)
            weights = torch.softmax(scores, dim=-1)
            r = (weights.unsqueeze(-1) * history_next).sum(dim=1)
            head_outputs.append(normalize_simplex(r))

        mix = torch.softmax(self.head_mix(hidden), dim=-1)  # (B, M)
        stacked = torch.stack(head_outputs, dim=1)  # (B, M, D)
        retrieved = (mix.unsqueeze(-1) * stacked).sum(dim=1)
        return normalize_simplex(retrieved)
