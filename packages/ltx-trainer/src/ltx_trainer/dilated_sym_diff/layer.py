"""Differentiable morphological layer stub — Sec. 4 extension."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dilated_sym_diff.config import DilatedSymDiffConfig
from ltx_trainer.dilated_sym_diff.morphology import dilated_symmetric_difference
from ltx_trainer.dilated_sym_diff.synthetic import die_face_mask, misalign_shift_dilate


def morphological_layer_stub(
    a_flat: list[float],
    b_flat: list[float],
    *,
    radius: int = 8,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Soft-threshold masks → dilated symmetric difference (CPU stub)."""
    side = int(len(a_flat) ** 0.5)
    a = [[1 if v >= threshold else 0 for v in a_flat[i * side : (i + 1) * side]] for i in range(side)]
    b = [[1 if v >= threshold else 0 for v in b_flat[i * side : (i + 1) * side]] for i in range(side)]
    import numpy as np

    aa = np.array(a, dtype=bool)
    bb = np.array(b, dtype=bool)
    diff = dilated_symmetric_difference(aa, bb, radius)
    return {
        "radius": radius,
        "diff_pixels": int(diff.sum()),
        "learnable_radius": False,
    }


def layer_card(cfg: DilatedSymDiffConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DilatedSymDiffConfig()
    a = die_face_mask()
    b = misalign_shift_dilate(a, dilate_r=3, dx=4, dy=2)
    import numpy as np

    diff = dilated_symmetric_difference(a, b, cfg.default_radius)
    return {
        "type": "binary_morphological_neural_network_stub",
        "learnable_radius": cfg.learnable_radius,
        "default_r": cfg.default_radius,
        "demo_diff_pixels": int(diff.sum()),
        "references": ["Aouad & Talbot BMNN", "Shen et al. Deep Morphological NN"],
    }
