"""Layer Fine-tuning Sensitivity (LFS) and layer-adaptive noise (Cert-LAS §4.4)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def average_l2_norm(delta: np.ndarray) -> float:
    """Definition 4.1: ||θ_l(t2)-θ_l(t1)||_2 / sqrt(d_l)."""
    d = max(1, int(delta.size))
    return float(np.linalg.norm(delta.ravel()) / math.sqrt(d))


def layer_fine_tuning_sensitivity(
    layer_deltas: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Definition 4.5: LFS(l) = δ̄_l / mean_j δ̄_j."""
    norms = {name: average_l2_norm(d) for name, d in layer_deltas.items()}
    if not norms:
        return {}
    mean_norm = sum(norms.values()) / len(norms)
    if mean_norm <= 0:
        return {k: 1.0 for k in norms}
    return {k: v / mean_norm for k, v in norms.items()}


def layer_dims(layer_deltas: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {k: max(1, int(v.size)) for k, v in layer_deltas.items()}


def allocate_layer_sigmas(
    lfs: Mapping[str, float],
    dims: Mapping[str, int],
    *,
    sigma_u: float,
) -> dict[str, float]:
    """
    Proposition 4.6: σ_l ∝ LFS(l) with budget equivalence (Definition 4.4).
    """
    if not lfs:
        return {}
    num = sum(float(dims.get(k, 1)) for k in lfs)
    den = sum(float(dims.get(k, 1)) * float(lfs[k]) ** 2 for k in lfs)
    if den <= 0 or num <= 0:
        return {k: sigma_u for k in lfs}
    scale = math.sqrt(num / den)
    return {k: sigma_u * float(lfs[k]) * scale for k in lfs}


def budget_equivalent_uniform_sigma(
    sigmas: Mapping[str, float],
    dims: Mapping[str, int],
) -> float:
    """σ_u^2 = sum(d_l σ_l^2) / sum(d_l)."""
    num = sum(float(dims[k]) * float(sigmas[k]) ** 2 for k in sigmas)
    den = sum(float(dims[k]) for k in sigmas)
    if den <= 0:
        return 0.0
    return math.sqrt(num / den)


def synthetic_unet_layer_deltas(seed: int = 0) -> dict[str, np.ndarray]:
    """Toy UNet-like layer blocks for smoke tests (compress-then-generate pattern)."""
    rng = np.random.default_rng(seed)
    spec = {
        "time_embedding.linear_1.bias": 128,
        "up_blocks.3.resnets.2.norm2.weight": 320,
        "up_blocks.2.resnets.0.norm2.weight": 320,
        "mid_block.resnets.1.norm2.weight": 640,
        "down_blocks.2.attentions.1.transformer_blocks.0.norm2.bias": 640,
    }
    out: dict[str, np.ndarray] = {}
    for name, d in spec.items():
        base = 0.02 if "time_embedding" in name else 0.15
        if "up_blocks.3" in name or "attn" in name:
            base *= 1.8
        out[name] = rng.standard_normal(d) * base
    return out
