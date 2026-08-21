"""Latent spin-glass phase diagnostics (Ascárate et al., arXiv:2606.02600)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CompressionMode = Literal["zero", "half", "full"]


@dataclass(frozen=True)
class HypersphericalPrior:
    """Angular compression / external-field direction (Fig. 1–3)."""

    mode: CompressionMode = "half"
    latent_dim: int = 128
    block_spin_dim: int = 3
    k_nn: int = 3


@dataclass(frozen=True)
class TrainingSchedule:
    """Cyclical annealing for KLD gain β (Methods / Appendix D)."""

    total_epochs: int = 300
    beta_warmup_epochs: int = 100
    beta_plateau: float = 1.0


@dataclass
class LatentPhaseConfig:
    paper_arxiv: str = "2606.02600"
    prior: HypersphericalPrior = field(default_factory=HypersphericalPrior)
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    lambda_prior: float = 1.0
    edge_alpha_star: float = 0.55
