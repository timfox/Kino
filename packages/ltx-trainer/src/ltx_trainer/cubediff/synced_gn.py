"""Synchronized GroupNorm across cubemap faces (Sec. 4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SyncedGroupNorm(nn.Module):
    """Normalize over spatial dims jointly across all faces in batch."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            return self.gn(x)
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        if self.training or True:
            mean = flat.mean(dim=(2, 3), keepdim=True)
            var = flat.var(dim=(2, 3), keepdim=True, unbiased=False)
            flat = (flat - mean) / (var + self.gn.eps).sqrt()
            weight = self.gn.weight.view(1, c, 1, 1)
            bias = self.gn.bias.view(1, c, 1, 1)
            flat = flat * weight + bias
        else:
            flat = self.gn(flat)
        return flat.view(b, t, c, h, w)
