"""RF volumetric compositing stub (Sec. 4.2, Eq. 3)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gsparc.compositing import rf_volumetric_composite


def render_spectrum_stub(
    emissions: Tensor,
    alphas: Tensor,
    *,
    height: int,
    width: int,
    use_distance_weight: bool = True,
    distance_scale: Tensor | None = None,
    depths: Tensor | None = None,
) -> Tensor:
    """
    Hemispherical spectrum via per-pixel Eq. 3 compositing (scatter stub).

    emissions: G×2; alphas: G; distance_scale: G (d_i); depths: G to receiver.
    """
    device = emissions.device
    dtype = emissions.dtype
    spec_r = torch.zeros(height, width, device=device, dtype=dtype)
    spec_i = torch.zeros(height, width, device=device, dtype=dtype)
    g = emissions.shape[0]
    if g == 0:
        return torch.stack([spec_r, spec_i], dim=0)

    if depths is None:
        depths = torch.arange(g, device=device, dtype=dtype)

    tx_dist = distance_scale if distance_scale is not None else torch.ones(g, device=device, dtype=dtype)

    for v in range(height):
        for u in range(width):
            # assign Gaussians to pixels by hash (different set per pixel)
            idx = ((torch.arange(g, device=device) + u * 7 + v * 13) % g).long()
            pix_alpha = alphas[idx]
            pix_emit = emissions[idx]
            pix_depth = depths[idx]
            pix_tx = tx_dist[idx]
            comp = rf_volumetric_composite(
                pix_alpha,
                pix_emit,
                pix_depth,
                pix_tx,
                use_distance_attenuation=use_distance_weight,
            )
            spec_r[v, u] = comp[0]
            spec_i[v, u] = comp[1]

    return torch.stack([spec_r, spec_i], dim=0)


def spectrum_to_channel(z_hat: Tensor) -> Tensor:
    """Eq. (2): ĥ = (1/HW) Σ_{u,v} ẑ[u,v] (complex mean)."""
    if z_hat.shape[0] == 2:
        real = z_hat[0].mean()
        imag = z_hat[1].mean()
        return torch.stack([real, imag])
    return z_hat.mean(dim=(-2, -1))
