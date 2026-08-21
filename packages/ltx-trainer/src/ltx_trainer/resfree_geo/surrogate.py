"""Surrogate blocks: EAC, mapping u = x + B⊙ψ (Sec. 4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def edge_aware_conv(f: Tensor) -> Tensor:
    """CEA(f) = w·C_edge + (1-w)·C_loc (Eq. 17) simplified."""
    if f.dim() == 3:
        f = f.unsqueeze(0)
    c = f.shape[1]
    local = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False).to(f.device)
    edge = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False).to(f.device)
    nn.init.dirac_(local.weight)
    nn.init.dirac_(edge.weight)
    loc = local(f)
    edg = edge(f)
    grad = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    pad = torch.zeros(f.shape[0], 1, 1, f.shape[3], device=f.device)
    weight = torch.sigmoid(torch.cat([grad, pad], dim=2))
    return weight * edg + (1.0 - weight) * loc


def apply_boundary_mask(displacement: Tensor, boundary_mask: Tensor) -> Tensor:
    """u = x + B(x) ⊙ ψ(x) (Eq. 18)."""
    return displacement * boundary_mask


def square_boundary_mask(h: int, w: int, alpha: float = 1.0) -> Tensor:
    """B(x) = α·x(1-x) per axis (Sec. 5.2)."""
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    bx = alpha * xx * (1 - xx)
    by = alpha * yy * (1 - yy)
    return torch.stack([bx, by], dim=0)


class TinyGeoSurrogate(nn.Module):
    """Minimal conv surrogate for smoke forward passes."""

    def __init__(self, in_ch: int, out_ch: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, out_ch, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) * 0.1


def forward_mapping(
    model: TinyGeoSurrogate,
    encoded: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Predict ψ and mapped coords."""
    if encoded.dim() == 3:
        encoded = encoded.unsqueeze(0)
    psi = edge_aware_conv(model(encoded))
    if psi.shape[0] == 1:
        disp = psi[0]
    else:
        disp = psi
    disp = apply_boundary_mask(disp, mask)
    h, w = mask.shape[-2], mask.shape[-1]
    xs = torch.linspace(0, 1, w, device=encoded.device)
    ys = torch.linspace(0, 1, h, device=encoded.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([xx, yy], dim=0)
    u = base + disp
    return u, disp
