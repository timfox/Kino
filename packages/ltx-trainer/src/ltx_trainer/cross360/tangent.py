"""Tangent projection sampling (Sec. III-C.1, gnomonic / OmniFusion-style)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cross360.config import NUM_TP_PATCHES_FULL, NUM_TP_PATCHES_INCOMPLETE, TP_FOV_DEG

# Full-sphere layout: latitudes −72°, −36°, 0°, 36°, 72° with 3,6,8,6,3 patches
TP_LATITUDES_FULL: tuple[float, ...] = (-72.0, -36.0, 0.0, 36.0, 72.0)
TP_PATCHES_PER_ROW_FULL: tuple[int, ...] = (3, 6, 8, 6, 3)

# Incomplete FOV (M3D / S2D3D): −31.2°, 0°, 31.2° with 6,8,6 patches
TP_LATITUDES_INCOMPLETE: tuple[float, ...] = (-31.2, 0.0, 31.2)
TP_PATCHES_PER_ROW_INCOMPLETE: tuple[int, ...] = (6, 8, 6)


def tp_sampling_layout(*, incomplete_fov: bool = False) -> dict[str, Any]:
    if incomplete_fov:
        return {
            "N": NUM_TP_PATCHES_INCOMPLETE,
            "latitudes_deg": list(TP_LATITUDES_INCOMPLETE),
            "patches_per_row": list(TP_PATCHES_PER_ROW_INCOMPLETE),
            "fov_deg": TP_FOV_DEG,
        }
    return {
        "N": NUM_TP_PATCHES_FULL,
        "latitudes_deg": list(TP_LATITUDES_FULL),
        "patches_per_row": list(TP_PATCHES_PER_ROW_FULL),
        "fov_deg": TP_FOV_DEG,
    }


def sample_tp_patches_from_erp(
    erp: Tensor,
    *,
    patch_size: int = 64,
    incomplete_fov: bool = False,
) -> Tensor:
    """Stub ERP→TP: N perspective crops as (B, N, C, patch_h, patch_w)."""
    if erp.dim() != 4:
        raise ValueError("erp must be B×C×H×W")
    b, c, h, w = erp.shape
    layout = tp_sampling_layout(incomplete_fov=incomplete_fov)
    n = layout["N"]
    patches: list[Tensor] = []
    for i in range(n):
        # non-uniform horizontal strip sampling (placeholder geometry)
        x0 = int((i / max(n, 1)) * max(w - patch_size, 1))
        y0 = int(h * 0.25)
        crop = erp[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if crop.shape[-2:] != (patch_size, patch_size):
            crop = F.interpolate(crop, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
        patches.append(crop)
    return torch.stack(patches, dim=1)


def tp_patches_to_erp_stub(patches: Tensor, erp_size: tuple[int, int]) -> Tensor:
    """Stub TP2ERP: scatter-mean patches back to ERP canvas."""
    b, n, c, ph, pw = patches.shape
    h, w = erp_size
    canvas = torch.zeros(b, c, h, w, device=patches.device, dtype=patches.dtype)
    count = torch.zeros(b, 1, h, w, device=patches.device, dtype=patches.dtype)
    for i in range(n):
        x0 = int((i / max(n, 1)) * max(w - pw, 1))
        y0 = int(h * 0.25)
        resized = F.interpolate(patches[:, i], size=(min(ph, h - y0), min(pw, w - x0)), mode="bilinear", align_corners=False)
        rh, rw = resized.shape[-2:]
        canvas[:, :, y0 : y0 + rh, x0 : x0 + rw] += resized
        count[:, :, y0 : y0 + rh, x0 : x0 + rw] += 1.0
    return canvas / count.clamp_min(1.0)
