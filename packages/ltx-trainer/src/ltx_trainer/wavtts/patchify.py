"""Waveform patchification (F samples per patch)."""

from __future__ import annotations

import torch
from torch import Tensor


def patchify_waveform(x: Tensor, patch_size: int) -> Tensor:
    """(B, T) or (B, 1, T) -> (B, N, F) with zero-pad tail."""
    if x.dim() == 3:
        x = x.squeeze(1)
    b, t = x.shape
    pad = (patch_size - t % patch_size) % patch_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    n = x.shape[-1] // patch_size
    return x.reshape(b, n, patch_size)


def unpatchify_waveform(patches: Tensor) -> Tensor:
    """(B, N, F) -> (B, T)."""
    b, n, f = patches.shape
    return patches.reshape(b, n * f)
