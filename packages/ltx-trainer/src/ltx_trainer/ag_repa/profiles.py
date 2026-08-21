"""Synthetic FoG-A / LASP profiles for smoke and demos."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.fog_a import FogAScore
from ltx_trainer.ag_repa.lasp import LayerScore


def synthetic_fog_profile(cfg: AgRepaConfig | None = None, *, seed: int = 0) -> list[FogAScore]:
    """Paper-shaped attribution: L1 dominant, mid-layer transition bump at L7."""
    c = cfg or AgRepaConfig()
    rng = np.random.default_rng(seed)
    scores: list[FogAScore] = []
    for layer in range(1, c.num_dit_layers + 1):
        if layer == 1:
            base = 0.167
        elif layer in (2, 7):
            base = 0.09 if layer == 2 else 0.085
        elif layer >= 20:
            base = 0.02
        else:
            base = 0.04
        scores.append(FogAScore(layer=layer, score=float(base + rng.normal(scale=0.005))))
    return scores


def synthetic_lasp_profile(cfg: AgRepaConfig | None = None, *, domain: str = "semantic") -> list[LayerScore]:
    c = cfg or AgRepaConfig()
    scores: list[LayerScore] = []
    for layer in range(1, c.num_dit_layers + 1):
        if domain == "semantic":
            base = 0.10 + 0.22 * (layer / c.num_dit_layers) ** 2
        else:
            base = 0.12 + 0.08 * np.sin(layer / 4.0)
        scores.append(LayerScore(layer=layer, score=float(base)))
    return scores
