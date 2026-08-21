"""Hyperparameters for CMAP (Mandalika, arXiv:2605.25708)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CMAPConfig:
    """Defaults match paper Sec. 3–4 (CLIP ViT-B/16, d=512); small dims allowed for tests."""

    embed_dim: int = 512
    """CLIP joint embedding dimension ``d``."""

    k_means_clusters: int = 3
    """K for per-class visual prototypes (Sec. 3.3)."""

    top_class_scores: int = 5
    """``k`` in Eq. (7) — mean over top-``k`` class confidences."""

    percentile_high: float = 0.8
    percentile_low: float = 0.2
    """Task-adaptive θ_up / θ_low from training confidences (Sec. 3.3)."""

    gumbel_temperature: float = 3.0
    """τ in Eq. (9)."""

    n_text_layers: int = 8
    """L_txt for symmetric gate parameter count (paper: 8)."""
