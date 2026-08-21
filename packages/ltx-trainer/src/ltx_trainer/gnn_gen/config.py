"""Configuration for GNN generalisation reference utilities (Ayday et al., arXiv:2605.25452)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GNNGenConfig:
    """Small defaults for tests; scale up for CORA-scale kernel studies."""

    embed_dim: int = 16
    n_gcn_layers: int = 2
    """Depth L in Theorems 1–3."""

    mc_samples: int = 256
    """Monte Carlo draws for ReLU kernel expectations when closed form is skipped."""

    wl_max_iters: int = 32
    """1-WL colour refinement cap (Section 2, Figure 2)."""

    csbm_feature_dim: int = 8
    """Node feature dimension d0 in CSBM(n, p, q, mu)."""
