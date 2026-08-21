"""Hyperparameters for F-RNG (Fu et al., arXiv:2605.25975)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FRNGConfig:
    """Defaults: small for unit tests. Paper / supplementary use larger dims (Sec. 8.2–8.3)."""

    token_dim: int = 64
    """Channel width for all token streams (LRM geometry/appearance, IDM prior, material)."""

    patch_size: int = 8
    """Spatial patch size (paper: 8×8 following RelitLRM)."""

    # Saliency (supplementary Sec. 7)
    box_kernel: int = 5
    gaussian_sigma: float = 2.0
    edge_attenuation: float = 6.0
    """Larger → stronger suppression near detected edges (boundary buffer proxy)."""

    saliency_top_frac: float = 0.02
    """Fraction of patches per view to refine (paper: top-2%)."""

    # Fine geometry (Sec. 4.1)
    fine_geo_hidden_mult: int = 4

    # MaterialFormer (supplementary Sec. 8.2: 4 layers, hidden 1024 — scaled down in tests)
    matformer_layers: int = 4
    matformer_dim_feedforward: int = 256
    matformer_heads: int = 4
    matformer_dropout: float = 0.0

    # Universal neural appearance decoder (Sec. 8.3: 4-layer MLP, 256 units)
    appearance_decoder_layers: int = 4
    appearance_decoder_width: int = 128
    appearance_out_dim: int = 3
    """Deferred-shading RGB (or linear radiance) per ray sample."""

    # Light-independence (Eq. 12); supplementary: λ_c = λ_KLD = 0.2
    lambda_cos: float = 0.2
    lambda_kld: float = 0.2
