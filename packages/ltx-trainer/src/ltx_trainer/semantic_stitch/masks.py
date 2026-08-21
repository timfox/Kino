"""Salient object union mask and dynamic mask optimization (Alg. 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.semantic_stitch.config import SemanticStitchConfig
from ltx_trainer.semantic_stitch.losses import composite_coverage_loss


def saliency_union(m1: NDArray[np.floating], m2: NDArray[np.floating], *, thresh: float = 0.5) -> NDArray[np.floating]:
    """O = M_t ∩ M_r style union of foreground masks."""
    return ((m1 > thresh) | (m2 > thresh)).astype(np.float64)


def optimize_seam_masks(
    object_mask: NDArray[np.floating],
    *,
    cfg: SemanticStitchConfig | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[dict[str, float]]]:
    """Area-based dynamic mask optimization (Algorithm 1)."""
    cfg = cfg or SemanticStitchConfig()
    rng = np.random.default_rng(seed)
    h, w = object_mask.shape
    l1 = rng.random((h, w))
    l2 = 1.0 - l1
    history: list[dict[str, float]] = []
    prev = float("inf")
    for _ in range(cfg.max_mask_epochs):
        losses = composite_coverage_loss(object_mask, l1, l2, cfg)
        history.append(losses)
        if abs(prev - losses["L_total"]) < 1e-5:
            break
        prev = losses["L_total"]
        # gradient-free nudge toward lower loss
        step = 0.05
        if losses["chosen_mask"] == 1.0:
            l1 = np.clip(l1 + step * object_mask, 0, 1)
        else:
            l2 = np.clip(l2 + step * object_mask, 0, 1)
        s = l1 + l2 + 1e-8
        l1, l2 = l1 / s, l2 / s
    return l1, l2, history
