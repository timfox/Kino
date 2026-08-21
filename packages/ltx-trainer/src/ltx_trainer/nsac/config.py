"""Hyperparameters for Neuronal Stochastic Attention Circuit (NSAC)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NSACConfig:
    """Defaults aligned with paper appendix B.2 (tabular / small-seq)."""

    d_model: int = 64
    n_heads: int = 16
    dropout: float = 0.0
    top_k: int = 8
    """Reserved for sparse NAC-style curation (Child et al.); full attention if L <= top_k or disabled."""
    sparsity: float = 0.5
    """Reserved for future block masking; currently unused in full-attention path."""
    n_mc: int = 5
    lambda_reg: float = 1.0
    """Weight on epistemic-separation term L_reg (Eq. 10)."""
    mu_pert: float = 0.0
    sigma_pert: float = 5.0
    """Fake OOD: x_ood = x_id + N(mu_pert, sigma_pert^2) (Eq. 8)."""
    reg_eps: float = 1e-8
    kappa_floor: float = 1e-4
    use_sparse_curation: bool = False
    """If True and L > top_k, keep only top_k keys per query by dot-product prescore (cheap routing)."""
