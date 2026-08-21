"""Fair CRPS estimators (Eq. 3–4, Zamo & Naveau 2018)."""

from __future__ import annotations

import torch
from torch import Tensor


def fair_crps(samples: Tensor, target: Tensor) -> Tensor:
    """Unbiased fair CRPS for scalar targets. ``samples``: ``(K,)``, ``target``: scalar."""
    if samples.dim() == 0:
        samples = samples.unsqueeze(0)
    k = samples.shape[0]
    target = target.reshape(())
    if k == 1:
        return (samples[0] - target).abs()
    rel = (samples - target).abs().mean()
    idx_i, idx_j = torch.triu_indices(k, k, offset=1)
    spread = (samples[idx_i] - samples[idx_j]).abs().mean()
    return rel - spread


def fair_crps_multivariate(samples: Tensor, target: Tensor) -> Tensor:
    """Per-entry fair CRPS averaged over dimensions (Eq. 4)."""
    if samples.dim() == 1:
        samples = samples.unsqueeze(-1)
        target = target.unsqueeze(-1)
    # samples: (K, D), target: (D,)
    k, d = samples.shape
    total = torch.zeros((), device=samples.device, dtype=samples.dtype)
    for dim in range(d):
        total = total + fair_crps(samples[:, dim], target[dim])
    return total / d
