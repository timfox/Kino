"""LoRA strength as continuous control (Eq. 7, Sec. 3.3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StrengthScaledLoRA(nn.Module):
    """W' = W + s · (A @ B) with frozen base W (ConceptSlider-style, training-time s)."""

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: int = 64) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.lora_a = nn.Linear(in_f, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / max(rank, 1)

    def forward(self, x: Tensor, strength: float | Tensor = 1.0) -> Tensor:
        s = strength if isinstance(strength, Tensor) else torch.tensor(strength, device=x.device, dtype=x.dtype)
        delta = self.lora_b(self.lora_a(x)) * self.scaling
        if isinstance(strength, Tensor) and strength.dim() > 0:
            while s.dim() < delta.dim():
                s = s.unsqueeze(-1)
            delta = delta * s
        else:
            delta = delta * float(s)
        return self.base(x) + delta
