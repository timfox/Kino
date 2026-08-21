"""BES configuration — search, backward blend, experiment presets."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.bes.constants import (
    GITHUB_REPO,
    OPERATOR_PROBS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PROJECT_PAGE,
)


@dataclass
class BESConfig:
    """Bidirectional Evolutionary Search hyperparameters."""

    name: str = "BES"
    paper_arxiv: str = PAPER_ARXIV
    title: str = PAPER_TITLE
    website: str = PROJECT_PAGE
    github: str = GITHUB_REPO

    # Search budget
    budget_calls: int = 200
    decompose_interval: int = 10
    k_max_expand: int = 3

    # Backward scoring (Eq. 5–6)
    alpha: float = 0.3

    # Boltzmann parent selection (Eq. 3–4)
    tau_start: float = 2.0
    tau_end: float = 1.0
    unexplored_bonus: float = 0.1

    # Operator probabilities (must sum to 1)
    operator_probs: dict[str, float] = field(default_factory=lambda: dict(OPERATOR_PROBS))

    # Post-training group size (paper: 8 trajectories per problem)
    group_size: int = 8


@dataclass
class KnightsKnavesPreset:
    """Table 5 — logical reasoning."""

    search_budget: int = 200
    decompose_interval: int = 10
    tau_start: float = 2.0
    tau_end: float = 1.0
    alpha: float = 0.3
    group_size: int = 8


@dataclass
class MuSiQuePreset:
    """Table 6 — multi-hop agent search."""

    search_budget: int = 50
    k_parallel: int = 4
    tau_start: float = 1.5
    tau_end: float = 0.3
    alpha: float = 0.7
    embed_threshold: float = 0.6
    group_size: int = 8


@dataclass
class OpenProblemPreset:
    """Table 7 — ShinkaEvolve + BES program evolution."""

    num_generations: int = 100
    stagnation_generations: int = 5
    stagnation_margin: float = 1e-2
    children_per_decompose: tuple[int, int] = (2, 4)
    max_tree_depth: int = 2
    alpha: float = 0.3
    api_budget_usd: float = 50.0
