"""LoRA adapters for lightweight DiT finetuning (<1% params, Sec. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def inject_lora(module: nn.Module, *, rank: int = 4, target: type = nn.Linear) -> list[LoRALinear]:
    """Replace ``target`` layers with LoRA-wrapped versions; returns adapter list."""
    adapters: list[LoRALinear] = []

    def _wrap(parent: nn.Module, name: str, child: nn.Module) -> None:
        if isinstance(child, target):
            wrapped = LoRALinear(child, rank=rank)
            setattr(parent, name, wrapped)
            adapters.append(wrapped)

    for name, child in list(module.named_children()):
        if isinstance(child, target):
            _wrap(module, name, child)
        else:
            adapters.extend(inject_lora(child, rank=rank, target=target))
    return adapters
