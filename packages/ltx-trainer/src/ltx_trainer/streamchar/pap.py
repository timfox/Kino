"""Progress-Aware Pointer (Eq. 5, Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProgressAwarePointer(nn.Module):
    """Predict spoken transcript endpoint ŝ from transcript states and audio condition c_a."""

    def __init__(self, dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.offset = nn.Parameter(torch.zeros(1))
        self.confidence = nn.Linear(dim, 1)

    def forward(
        self,
        transcript_states: Tensor,
        audio_condition: Tensor,
        transcript_len: int,
    ) -> Tensor:
        """transcript_states (B, N, D), audio_condition (B, T, D) -> endpoint index in [0, N]."""
        b, n, d = transcript_states.shape
        q = audio_condition
        k = v = transcript_states
        attn_out, _ = self.cross(q, k, v)
        wj = torch.softmax(self.confidence(attn_out).squeeze(-1), dim=1)
        positions = torch.arange(q.shape[1], device=q.device, dtype=attn_out.dtype)
        s_hat = (wj * positions.unsqueeze(0)).sum(dim=1) + self.offset.squeeze()
        return s_hat.clamp(0, float(transcript_len))
