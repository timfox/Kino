"""TartanAir cubemap → ERP panorama stub (Sec. 3.1.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def cubemap_to_erp_stub(
    faces: Tensor | None = None,
    *,
    height: int = 64,
    width: int = 128,
    num_faces: int = 6,
) -> Tensor:
    """Synthetic ERP from six perspective faces (B, 6, 3, Hf, Wf) or random."""
    if faces is None:
        faces = torch.rand(1, num_faces, 3, height // 4, width // 4)
    b = faces.shape[0]
    pano = torch.zeros(b, 3, height, width, device=faces.device, dtype=faces.dtype)
    stripe = max(1, width // num_faces)
    for i in range(num_faces):
        face_rgb = faces[:, i].mean(dim=(-2, -1), keepdim=True).expand(-1, -1, height, stripe)
        pano[:, :, :, i * stripe : (i + 1) * stripe] = face_rgb
    return pano
