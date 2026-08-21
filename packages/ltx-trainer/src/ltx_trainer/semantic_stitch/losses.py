"""Composite Coverage Loss: L_comp + L_excl + L_smooth (Eq. 2–5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.semantic_stitch.config import SemanticStitchConfig


def mask_area(mask: NDArray[np.floating]) -> float:
    return float(mask.sum())


def object_completeness_loss(
    object_mask: NDArray[np.floating],
    l1: NDArray[np.floating],
    l2: NDArray[np.floating],
) -> tuple[float, int]:
    """L_comp with dynamic k = argmax area(O ⊙ L_k) (Eq. 2)."""
    m1 = object_mask * l1
    m2 = object_mask * l2
    a1, a2 = mask_area(m1), mask_area(m2)
    if a1 >= a2:
        diff = object_mask - m1
    else:
        diff = object_mask - m2
    n = max(float(object_mask.sum()), 1.0)
    return float(np.mean(diff**2)), int(a1 >= a2)


def exclusivity_loss(l2: NDArray[np.floating], object_mask: NDArray[np.floating]) -> float:
    """L_excl = Σ M2² where M2 = O ⊙ L2 (Eq. 3)."""
    m2 = object_mask * l2
    return float(np.mean(m2**2))


def smoothness_loss(mask: NDArray[np.floating]) -> float:
    """L_smooth via squared gradients (Eq. 4)."""
    gx = np.diff(mask, axis=1, prepend=mask[:, :1])
    gy = np.diff(mask, axis=0, prepend=mask[:1, :])
    return float(np.mean(gx**2 + gy**2))


def composite_coverage_loss(
    object_mask: NDArray[np.floating],
    l1: NDArray[np.floating],
    l2: NDArray[np.floating],
    cfg: SemanticStitchConfig | None = None,
) -> dict[str, float]:
    """L_total = L_comp + L_excl + L_smooth (Eq. 5)."""
    cfg = cfg or SemanticStitchConfig()
    l_comp, chosen = object_completeness_loss(object_mask, l1, l2)
    l_excl = exclusivity_loss(l2, object_mask) * cfg.exclusivity_weight
    l_smooth = (smoothness_loss(l1) + smoothness_loss(l2)) * cfg.smooth_weight
    return {
        "L_comp": l_comp,
        "L_excl": l_excl,
        "L_smooth": l_smooth,
        "L_total": l_comp + l_excl + l_smooth,
        "chosen_mask": float(chosen),
    }
