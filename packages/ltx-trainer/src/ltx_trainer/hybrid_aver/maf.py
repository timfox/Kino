"""Modality-temporal attention (Eqs. 11–12)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ModalityTemporalAttention(nn.Module):
    """Weighted aggregation over 2T multimodal tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.W = nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, 2T, D)
        e = torch.tanh(self.W(tokens) + self.b)
        scores = self.w(e).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        return torch.sum(alpha.unsqueeze(-1) * tokens, dim=1)
