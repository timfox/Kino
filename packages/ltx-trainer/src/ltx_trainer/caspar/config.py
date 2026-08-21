"""Caspar configuration (arXiv:2605.30583)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CasparConfig:
    """GPU symbolic nonlinear solver stub aligned with SymForce/Caspar."""

    name: str = "Caspar"
    paper_arxiv: str = "arXiv:2605.30583"
    title: str = "CUDA Accelerator for Symbolic Programming with Adaptive Reordering"
    upstream: str = "https://github.com/symforce-org/symforce"

    # LM / PCGNR (Madsen et al.; Wu et al. no Schur)
    initial_trust_region: float = 100.0
    pcg_tolerance: float = 1e-3
    max_lm_iterations: int = 50
    max_pcg_iterations: int = 200

    # Symbolic codegen
    use_blocked_soa: bool = True
    vector_chunk_size: int = 4
    enable_partial_cse: bool = True
    enable_context_optimizations: bool = True

    # BAL smoke problem sizes (paper Fig. 3 subsets)
    bal_smoke_cameras: int = 4
    bal_smoke_points: int = 20
    bal_smoke_factors: int = 80
