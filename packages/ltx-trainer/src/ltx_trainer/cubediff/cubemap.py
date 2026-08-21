"""Cubemap face layout and ERP unfold stub."""

from __future__ import annotations

import torch
from torch import Tensor

FACE_NAMES = ("front", "right", "back", "left", "up", "down")


def erp_to_cubemap_faces(erp: Tensor, face_h: int, face_w: int) -> Tensor:
    """Project ERP batch to 6 perspective faces (B, 6, C, H, W) stub."""
    b, c, eh, ew = erp.shape
    faces = []
    for i in range(6):
        y0 = (i * eh) // 6
        y1 = ((i + 1) * eh) // 6
        patch = torch.nn.functional.interpolate(
            erp[:, :, y0:y1], size=(face_h, face_w), mode="bilinear", align_corners=False
        )
        faces.append(patch)
    return torch.stack(faces, dim=1)


def cubemap_to_erp(faces: Tensor, erp_h: int, erp_w: int) -> Tensor:
    """Stitch 6 faces to ERP (B, C, H, W) stub."""
    b, t, c, fh, fw = faces.shape
    strips = [faces[:, i] for i in range(t)]
    strip_h = erp_h // t
    parts = [
        torch.nn.functional.interpolate(s, size=(strip_h, erp_w), mode="bilinear", align_corners=False)
        for s in strips
    ]
    return torch.cat(parts, dim=-2)


def crop_face_overlap(faces: Tensor, *, train_fov: float, crop_fov: float) -> Tensor:
    """Crop 95° faces to central 90° (Sec. 4.4)."""
    if not train_fov > crop_fov:
        return faces
    ratio = crop_fov / train_fov
    b, t, c, h, w = faces.shape
    nh, nw = max(1, int(h * ratio)), max(1, int(w * ratio))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return faces[:, :, :, y0 : y0 + nh, x0 : x0 + nw]
