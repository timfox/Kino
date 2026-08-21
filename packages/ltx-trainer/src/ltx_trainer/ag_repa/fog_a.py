"""Forward-only Gate Ablation — FoG-A (paper Eq. 7–9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.ag_repa.config import AgRepaConfig


@dataclass(frozen=True)
class FogAScore:
    layer: int
    score: float


def fog_a_score(
    velocity_full: np.ndarray,
    velocity_ablated: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Normalized velocity perturbation when layer gate is closed (Eq. 9)."""
    num = np.linalg.norm(velocity_ablated.astype(np.float64) - velocity_full.astype(np.float64)) ** 2
    den = np.linalg.norm(velocity_full.astype(np.float64)) ** 2 + epsilon
    return float(num / den)


def fog_a_profile(
    *,
    velocity_full: np.ndarray,
    ablated_by_layer: dict[int, np.ndarray],
    cfg: AgRepaConfig | None = None,
) -> list[FogAScore]:
    c = cfg or AgRepaConfig()
    return [
        FogAScore(
            layer=layer,
            score=fog_a_score(velocity_full, v_abl, epsilon=c.fog_epsilon),
        )
        for layer, v_abl in sorted(ablated_by_layer.items())
    ]


def top_k_by_fog_a(scores: list[FogAScore], k: int) -> list[int]:
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    return [s.layer for s in ranked[:k]]


def attribution_weights(scores: list[FogAScore], selected_layers: list[int]) -> dict[int, float]:
    """λ_k proportional to FoG-A on selected set (Eq. 11)."""
    by_layer = {s.layer: s.score for s in scores}
    total = sum(by_layer[l] for l in selected_layers) + 1e-9
    return {l: by_layer[l] / total for l in selected_layers}
