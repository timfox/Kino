"""CIE L*a*b* conversion for ELC loss."""

from __future__ import annotations

import torch
from torch import Tensor

_D65 = (0.95047, 1.0, 1.08883)


def _linearize_srgb(x: Tensor) -> Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)).pow(2.4))


def rgb_to_lab(rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """sRGB ``[B,3,H,W]`` or ``[3,H,W]`` in ``[0,1]`` → L*, a*, b*."""
    squeeze = False
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
        squeeze = True
    lin = _linearize_srgb(rgb.clamp(0.0, 1.0))
    m = lin.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum("bchw,dc->bdhw", lin, m)
    xyz[..., 0, :, :] /= _D65[0]
    xyz[..., 1, :, :] /= _D65[1]
    xyz[..., 2, :, :] /= _D65[2]
    eps = 216.0 / 24389.0
    k = 24389.0 / 27.0
    f = torch.where(xyz > eps, xyz.pow(1.0 / 3.0), (k * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    if squeeze:
        return L.squeeze(0), a.squeeze(0), b.squeeze(0)
    return L, a, b
