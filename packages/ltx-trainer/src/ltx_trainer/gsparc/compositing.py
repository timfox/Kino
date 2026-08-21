"""RF alpha compositing with transmittance (Sec. 4.2, Eq. 3)."""

from __future__ import annotations

import torch
from torch import Tensor


def rf_volumetric_composite(
    alphas: Tensor,
    emissions: Tensor,
    depths: Tensor,
    tx_distance: Tensor,
    *,
    use_distance_attenuation: bool = True,
) -> Tensor:
    """
    S(θ,φ) = Σ_i [ α_i Π_{j<i}(1−α_j) s_i / d_i ]  (complex as 2-vector).

    alphas: G; emissions: G×2; depths: G (sort key); tx_distance: G (= d_i).
    """
    if alphas.numel() == 0:
        return torch.zeros(2, device=alphas.device, dtype=alphas.dtype)

    order = torch.argsort(depths, descending=False)
    transmittance = torch.tensor(1.0, device=alphas.device, dtype=alphas.dtype)
    accum = torch.zeros(2, device=alphas.device, dtype=alphas.dtype)

    for idx in order:
        a = alphas[idx].clamp(0.0, 1.0)
        s = emissions[idx]
        if use_distance_attenuation:
            s = s / tx_distance[idx].clamp(min=1e-3)
        accum = accum + transmittance * a * s
        transmittance = transmittance * (1.0 - a)

    return accum
