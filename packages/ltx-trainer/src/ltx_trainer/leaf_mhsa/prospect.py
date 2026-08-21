"""PROSPECT-PRO forward-mode baseline stub (§3)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.leaf_mhsa.config import LeafMHSAConfig


def prospect_pro_forward(traits: Tensor, n_bands: int, *, seed: int = 0) -> Tensor:
    """Generalized RTM stub: higher error in NIR/SWIR vs grapevine-specific MHSA."""
    g = torch.Generator(device=traits.device)
    g.manual_seed(seed)
    b = traits.shape[0]
    wl = torch.linspace(400, 2500, n_bands, device=traits.device)
    base = 0.08 + 0.35 * torch.sigmoid(traits[:, 15:16] - traits[:, 15:16].mean())
    spec = base.expand(b, n_bands).clone()
    nir = (wl >= 700) & (wl <= 1300)
    swir = (wl >= 1500) & (wl <= 2400)
    spec[:, nir] += 0.12 * torch.randn(b, int(nir.sum()), generator=g, device=traits.device)
    spec[:, swir] += 0.18 * torch.randn(b, int(swir.sum()), generator=g, device=traits.device)
    spec[:, ~(nir | swir)] += 0.05 * torch.randn(
        b, int((~(nir | swir)).sum()), generator=g, device=traits.device
    )
    return spec.clamp(0.0, 1.0)


def mae_vs_measured(
    predicted: Tensor,
    measured: Tensor,
    cfg: LeafMHSAConfig | None = None,
) -> Tensor:
    cfg = cfg or LeafMHSAConfig()
    _ = cfg
    return torch.mean(torch.abs(predicted - measured), dim=0)
