"""Three structural domain gaps (Sec. 3.1, Fig. 1)."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from torch import Tensor


class DomainGap(str, Enum):
    GEOMETRIC_DISTORTION = "geometric_distortion"
    NON_UNIFORM_SAMPLING = "non_uniform_sampling"
    BOUNDARY_CONTINUITY = "boundary_continuity"


GAP_DESCRIPTIONS: dict[DomainGap, str] = {
    DomainGap.GEOMETRIC_DISTORTION: "ERP pole stretching breaks translation-equivariant CNN assumptions.",
    DomainGap.NON_UNIFORM_SAMPLING: "Latitude-dependent pixel density (dense equator, sparse poles).",
    DomainGap.BOUNDARY_CONTINUITY: "Left/right ERP seam is adjacent on the sphere but disjoint in planar conv.",
}


def erp_latitude_weight(height: int, device: torch.device | None = None) -> Tensor:
    """Cos(latitude) weighting proxy for non-uniform spherical sampling (WS-PSNR family)."""
    v = torch.arange(height, device=device, dtype=torch.float32)
    theta = (v + 0.5) / height * torch.pi
    return torch.cos(theta - torch.pi / 2).clamp(min=1e-6)


def gap_severity_map(height: int, width: int) -> dict[str, Tensor]:
    """Synthetic maps illustrating relative severity of each gap on ERP lattice."""
    w_lat = erp_latitude_weight(height)
    distortion = (1.0 / w_lat.view(-1, 1).expand(height, width)).clamp(max=8.0)
    sampling = w_lat.view(-1, 1).expand(height, width)
    seam = torch.zeros(height, width)
    seam[:, : max(1, width // 32)] = 1.0
    seam[:, -max(1, width // 32) :] = 1.0
    return {
        DomainGap.GEOMETRIC_DISTORTION.value: distortion / distortion.max(),
        DomainGap.NON_UNIFORM_SAMPLING.value: 1.0 - sampling,
        DomainGap.BOUNDARY_CONTINUITY.value: seam,
    }


def gaps_card() -> dict[str, Any]:
    return {
        "gaps": [g.value for g in DomainGap],
        "descriptions": {g.value: GAP_DESCRIPTIONS[g] for g in DomainGap},
    }
