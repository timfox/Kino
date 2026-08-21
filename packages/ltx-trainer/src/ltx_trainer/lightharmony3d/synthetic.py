"""Synthetic LH3D-Bench-style scenes for stub training/eval."""

from __future__ import annotations

import torch
from torch import Tensor


def synthesize_scene(size: int = 64, *, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    bg = torch.rand(3, size, size, generator=g) * 0.4 + 0.2
    # window highlight
    bg[:, size // 4 : size // 2, size // 3 : 2 * size // 3] += 0.35
    return bg.clamp(0, 1)


def synthesize_object(size: int = 64, *, seed: int = 1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(3, size, size, generator=g) * 0.5 + 0.3
    mask = torch.zeros(1, size, size)
    cx, cy, r = size // 2, size // 2, size // 5
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    mask[0] = ((xx - cx) ** 2 + (yy - cy) ** 2 < r**2).float()
    return rgb, mask


def synthesize_insertion_pair(size: int = 64, *, seed: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """background, object_rgb, mask, pseudo ground-truth composite."""
    scene = synthesize_scene(size, seed=seed)
    obj_rgb, mask = synthesize_object(size, seed=seed + 17)
    from ltx_trainer.lightharmony3d.model import LightHarmony3D

    model = LightHarmony3D()
    model.eval()
    with torch.no_grad():
        out = model(scene.unsqueeze(0), obj_rgb.unsqueeze(0), mask.unsqueeze(0))
    gt = out.composite.squeeze(0)
    return scene, obj_rgb, mask, gt
