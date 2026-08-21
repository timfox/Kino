"""IntPhys benchmark stubs (Sec. 4.1)."""

from __future__ import annotations

import torch
from torch import Tensor

INTPHYS_BLOCK_DESCRIPTIONS: dict[str, str] = {
    "O1": "Object permanence — occluded objects reappear consistently",
    "O2": "Object continuity — no teleportation through occluders",
    "O3": "Object solidity — solids cannot occupy the same space",
}


def synthetic_intphys_features(
    n: int,
    dim: int,
    *,
    seed: int = 0,
    block: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Separable possible/impossible clusters for probe smoke (not real IntPhys video).

    Returns ``(features, labels, block_ids)`` with labels 0=possible, 1=impossible.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    blocks = torch.randint(0, 3, (n,), generator=g)
    if block is not None:
        bid = {"O1": 0, "O2": 1, "O3": 2}[block]
        blocks = torch.full((n,), bid, dtype=torch.long)

    direction = torch.randn(dim, generator=g)
    direction = direction / direction.norm()
    noise = torch.randn(n, dim, generator=g) * 0.3
    sign = labels.float().mul(2).sub(1)
    features = sign.unsqueeze(-1) * direction + noise
    block_offset = torch.nn.functional.one_hot(blocks, 3).float() @ torch.randn(3, dim, generator=g) * 0.2
    features = features + block_offset
    return features, labels, blocks
