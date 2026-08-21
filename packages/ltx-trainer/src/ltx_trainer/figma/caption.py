"""Caption-length saturation analysis (§3, Figure 2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.figma.config import FigmaConfig


def saturating_recall_curve(
    token_counts: np.ndarray,
    *,
    plateau_tokens: int = 50,
    r1_max: float = 20.81,
    r5_max: float = 47.71,
    r10_max: float = 62.94,
) -> dict[str, np.ndarray]:
    """Toy MuQ-MuLaN-style saturation beyond ~40–50 tokens (Figure 2)."""
    x = np.asarray(token_counts, dtype=np.float64)
    # Steep rise through ``plateau_tokens``, then flat (additional tokens ignored).
    norm = np.minimum(x, float(plateau_tokens)) / float(plateau_tokens)
    scale = (1.0 - np.exp(-3.0 * norm)) / (1.0 - np.exp(-3.0))
    return {
        "r1": r1_max * scale,
        "r5": r5_max * scale,
        "r10": r10_max * scale,
    }


def caption_saturation_demo(cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    tokens = np.arange(5, 105, 5)
    curve = saturating_recall_curve(tokens, plateau_tokens=c.caption_saturation_tokens)
    at_20 = float(curve["r1"][tokens.tolist().index(20)])
    at_50 = float(curve["r1"][tokens.tolist().index(50)])
    at_100 = float(curve["r1"][-1])
    return {
        "token_counts": tokens.tolist(),
        "r1": curve["r1"].tolist(),
        "plateau_tokens": c.caption_saturation_tokens,
        "delta_r1_20_to_50": at_50 - at_20,
        "delta_r1_50_to_100": at_100 - at_50,
        "saturates_after_50": abs(at_100 - at_50) < 0.5,
    }
