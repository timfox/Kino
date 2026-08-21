"""ERP / CMP projection helpers (Sec. 3.1, Fig. 2–3)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.oma_fme.config import CMP_FACE_SIZE, CMP_FACES
from ltx_trainer.oma_fme.variants import CMP_FACE_NAMES, ProjectionFormat


def erp_uv_grid(height: int, width: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Normalized ERP coordinates u∈[0,1), v∈[0,1)."""
    v = torch.linspace(0, 1, height, device=device, dtype=torch.float32)
    u = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu, vv


def erp_to_sphere_direction(u: Tensor, v: Tensor) -> Tensor:
    """Unit directions ω_ERP(u,v) for equirectangular sampling (Eq. 2)."""
    lon = (u - 0.5) * 2.0 * math.pi
    lat = (0.5 - v) * math.pi
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    return torch.stack([x, y, z], dim=-1)


def stub_erp_to_cmp_faces(erp: Tensor, face_size: int = 2048) -> dict[str, Tensor]:
    """Stub ERP→CMP: downsample ERP to six face tensors (not geometrically exact)."""
    if erp.dim() != 4:
        raise ValueError("erp must be B×C×H×W")
    b, c, h, w = erp.shape
    faces: dict[str, Tensor] = {}
    for i, name in enumerate(CMP_FACE_NAMES):
        # partition width into 6 strips as placeholder
        strip_w = w // CMP_FACES
        x0 = i * strip_w
        x1 = x0 + strip_w
        patch = erp[:, :, :, x0:x1]
        faces[name] = torch.nn.functional.interpolate(
            patch, size=(face_size, face_size), mode="bilinear", align_corners=False
        )
    return faces


def cmp_tile_count(projection: ProjectionFormat) -> int:
    return CMP_FACES if projection == ProjectionFormat.CMP else 1


def projection_metadata(projection: ProjectionFormat) -> dict[str, Any]:
    if projection == ProjectionFormat.ERP:
        return {"omaf_projection": "erp", "tiles": 1}
    return {
        "omaf_projection": "cmp",
        "tiles": CMP_FACES,
        "face_names": list(CMP_FACE_NAMES),
        "face_size": list(CMP_FACE_SIZE),
    }
