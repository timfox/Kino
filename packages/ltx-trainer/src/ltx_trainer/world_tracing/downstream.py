"""Training-free downstream pipelines (Sec. 5)."""

from __future__ import annotations

import torch
from torch import Tensor


def voxelize_point_stack(x: Tensor, *, voxel_size: float = 0.02) -> dict[str, Tensor | int]:
    """Voxelize multilayer XYZ for TRELLIS Stage-1 sparse structure."""
    pts = x.reshape(-1, 3)
    vox = torch.floor(pts / voxel_size).to(torch.int64)
    uniq = torch.unique(vox, dim=0)
    return {"voxels": uniq, "count": int(uniq.shape[0]), "voxel_size": voxel_size}


def compose_scene_edit(
    scene: Tensor,
    obj: Tensor,
    edit_mask: Tensor,
) -> Tensor:
    """Closed-form object insertion in shared camera frame (Sec. 5.1)."""
    mask = edit_mask > 0.5
    if scene.ndim == 4 and mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(scene)
    else:
        while mask.ndim < scene.ndim:
            mask = mask.unsqueeze(0 if mask.ndim == 2 else -1)
        mask = mask.expand_as(scene)
    return torch.where(mask, obj, scene)


def rasterize_layer_depth(
    x: Tensor,
    layer: int = 0,
) -> Tensor:
    """Depth track from multilayer stack for geometry-guided video (Sec. 5.2)."""
    if x.ndim == 5:
        depth = x[:, layer, ..., 2]
    else:
        depth = x[layer, ..., 2]
    return depth
