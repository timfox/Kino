"""Gram matrix for style statistics (Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def gram_matrix(features: Tensor) -> Tensor:
    """Compute G^l from activations F^l with shape N×M (channels × spatial)."""
    if features.dim() != 3:
        raise ValueError(f"expected B×C×HW features, got {features.shape}")
    b, c, hw = features.shape
    f = features.reshape(b, c, hw)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / float(c * hw)
