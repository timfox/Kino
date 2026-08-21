"""CLIP-style EEG–CLAP alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_contrastive_loss(z_eeg: Tensor, z_audio: Tensor, *, logit_scale: float | None = None) -> Tensor:
    """Symmetric cross-entropy over batch similarity (Sec. 3.2)."""
    z_eeg = F.normalize(z_eeg, dim=-1)
    z_audio = F.normalize(z_audio, dim=-1)
    scale = logit_scale if logit_scale is not None else 1.0 / 0.07
    logits = scale * z_eeg @ z_audio.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def ridge_to_clap(z_eeg: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Lightweight ridge adapter for AudioLDM conditioning (Fig. 2)."""
    out = z_eeg @ weight.T
    if bias is not None:
        out = out + bias
    return out
