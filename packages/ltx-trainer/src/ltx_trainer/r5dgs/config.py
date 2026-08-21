"""Hyperparameters for R5DGS (Gridusov et al., arXiv:2605.25909)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class R5DGSConfig:
    """Paper defaults where noted; small ``num_classes`` / dims for tests."""

    identity_dim: int = 16
    """Per-Gaussian identity encoding dimension (Sec. II-A)."""

    num_classes: int = 8
    """Object / instance classes for 2D identity supervision."""

    k_neighbors: int = 5
    """k-NN for L_3d and L_major (Sec. II-C)."""

    lambda_obj: float = 1.0
    lambda_3d: float = 2.0
    lambda_rigid: float = 0.5
    lambda_maj: float = 0.5
    """Eq. (2) and Sec. II-C."""

    tau_reg: int = 2
    """Apply L_3d / L_major every ``tau_reg`` steps (paper: every 2 iters)."""

    t_rigid: int = 10_000
    """Activate L_rigid after this iteration (paper: 10,000)."""

    # Lookup (Sec. II-D) — embedding dim is model-dependent; PE often 1024+
    lookup_embed_dim: int = 64
    """Stored per-group embedding size for cosine retrieval (tests)."""
