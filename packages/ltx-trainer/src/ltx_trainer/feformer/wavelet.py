"""Haar DWT / IDWT stubs for WAFF (Sec. 3.5, Eq. 25–30)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

SUBBAND_NAMES = ("LLL", "LLH", "LHL", "HLL", "LHH", "HLH", "HHL", "HHH")


def _down(x: Tensor) -> Tensor:
    return F.avg_pool3d(x, kernel_size=2, stride=2)


def _high(x: Tensor) -> Tensor:
    low = _down(x)
    up = F.interpolate(low, size=x.shape[-3:], mode="trilinear", align_corners=False)
    return x - up


def dwt3(x: Tensor) -> dict[str, Tensor]:
    """Approximate 3D Haar DWT into eight named sub-bands."""
    if min(x.shape[-3:]) < 4:
        half = _down(x)
        quarter = _down(half)
        eigth = _down(quarter)
        return {
            "LLL": eigth,
            "LLH": quarter - F.interpolate(eigth, size=quarter.shape[-3:], mode="trilinear", align_corners=False),
            "LHL": half - F.interpolate(quarter, size=half.shape[-3:], mode="trilinear", align_corners=False),
            "HLL": _high(half),
            "LHH": _high(quarter),
            "HLH": quarter * 0.5,
            "HHL": half * 0.5,
            "HHH": _high(x),
        }
    lll = _down(_down(_down(x)))
    llh = _down(_down(_high(x)))
    lhl = _down(_high(_down(x)))
    hll = _high(_down(_down(x)))
    return {
        "LLL": lll,
        "LLH": llh,
        "LHL": lhl,
        "HLL": hll,
        "LHH": _high(_down(x)),
        "HLH": _high(_high(_down(x))),
        "HHL": _high(_high(x)),
        "HHH": _high(_high(_high(x))),
    }


def idwt3(bands: dict[str, Tensor], *, target_shape: tuple[int, int, int] | None = None) -> Tensor:
    """Reconstruct spatial tensor from sub-bands (approximate inverse)."""
    core = bands.get("LLL")
    if core is None:
        raise ValueError("LLL band required")
    out = core
    for name in SUBBAND_NAMES[1:]:
        if name in bands:
            b = bands[name]
            out = out + F.interpolate(b, size=core.shape[-3:], mode="trilinear", align_corners=False) * 0.125
    if target_shape is not None:
        out = F.interpolate(out, size=target_shape, mode="trilinear", align_corners=False)
    return out
