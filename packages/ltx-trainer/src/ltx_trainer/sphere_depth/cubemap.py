"""Cubemap face extraction / reprojection stub (Sec. 3.2, Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

FACE_NAMES = ("front", "right", "back", "left", "top", "bottom")


def erp_to_cubemap_faces(erp: Tensor, face_size: int) -> dict[str, Tensor]:
    """Six perspective crops via resize stub (production: tangent-plane sampling)."""
    b, c, _, _ = erp.shape
    faces: dict[str, Tensor] = {}
    quarters = [
        erp[..., :, : erp.shape[-1] // 2],
        erp[..., :, erp.shape[-1] // 2 :],
        erp[..., : erp.shape[-2] // 2, :],
        erp[..., erp.shape[-2] // 2 :, :],
    ]
    for i, name in enumerate(FACE_NAMES):
        src = quarters[i % len(quarters)]
        faces[name] = F.interpolate(src, size=(face_size, face_size), mode="bilinear", align_corners=False)
    return faces


def _face_to_nchw(depth: Tensor) -> Tensor:
    if depth.dim() == 2:
        return depth.unsqueeze(0).unsqueeze(0)
    if depth.dim() == 3:
        return depth.unsqueeze(1)
    return depth


def cubemap_faces_to_erp(faces: dict[str, Tensor], height: int, width: int) -> Tensor:
    """Stitch face depth maps back to ERP (smoke: horizontal concat + resize)."""
    parts = [_face_to_nchw(faces[n]) for n in FACE_NAMES[:4]]
    stacked = torch.cat(parts, dim=-1)
    out = F.interpolate(stacked, size=(height, width), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0)
