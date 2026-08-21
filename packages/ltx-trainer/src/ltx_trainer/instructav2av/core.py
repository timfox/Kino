"""Backward-compatible SIGA helper."""

from __future__ import annotations

import torch

from ltx_trainer.instructav2av.siga import SIGAModule


def siga_gate(
    source: torch.Tensor,
    instruction: torch.Tensor,
    *,
    dim: int | None = None,
) -> torch.Tensor:
    """Apply SIGA; returns per-token gate G (Eq. 6)."""
    if source.dim() == 2:
        source = source.unsqueeze(0)
        instruction = instruction.unsqueeze(0)
    d = dim or source.shape[-1]
    mod = SIGAModule(d)
    fh = source
    fused, gate = mod(fh, source, instruction)
    _ = fused
    return gate
