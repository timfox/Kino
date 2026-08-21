"""Mesh convolution / pooling stubs (Eq. 3, Fig. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

_FAF_CACHE: dict[int, Tensor] = {}


def faf_neighbors(n_tri: int, *, mr: int, use_cache: bool) -> Tensor:
    """Face-adjacent-face indices (cyclic stub)."""
    key = (mr << 16) | n_tri
    if use_cache and key in _FAF_CACHE:
        return _FAF_CACHE[key]
    nbr = torch.stack(
        [
            torch.arange(n_tri, device="cpu") % n_tri,
            (torch.arange(n_tri) + 1) % n_tri,
            (torch.arange(n_tri) + 2) % n_tri,
        ],
        dim=0,
    )
    if use_cache:
        _FAF_CACHE[key] = nbr
    return nbr


class MeshConvStub(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(4, ch) / 4)
        self.b = nn.Parameter(torch.zeros(ch))

    def forward(self, f: Tensor, nbr: Tensor) -> Tensor:
        # f: B x N x C
        f0 = f
        f1 = f[:, nbr[0] % f.shape[1]]
        f2 = f[:, nbr[1] % f.shape[1]]
        f3 = f[:, nbr[2] % f.shape[1]]
        stack = torch.stack([f0, f1, f2, f3], dim=2)
        return (stack * self.w.view(1, 1, 4, -1)).sum(dim=2) + self.b


class MeshPoolStub(nn.Module):
    def forward(self, f: Tensor) -> Tensor:
        n = f.shape[1]
        if n < 4:
            return f
        return f[:, : n // 4 * 4].view(f.shape[0], -1, 4, f.shape[-1]).max(dim=2).values
