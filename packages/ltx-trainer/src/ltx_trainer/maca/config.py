"""Configuration for MACA (Multi-Agent Coordination Adaptation, arXiv:2605.25746)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MACAConfig:
    paper_arxiv: str = "arXiv:2605.25746"
    github: str = "https://github.com/However-Li/MACA"

    # Reference agent pool sizes used in the paper figures (code / math / QA).
    default_agent_pool: tuple[str, ...] = (
        "TaskPlanner",
        "AlgorithmDesigner",
        "CodeWriting",
        "CodeReviewer",
        "UnitTestWriter",
        "EdgeCaseHunter",
        "BugFixer",
        "Summarizer",
        "BudgetController",
        "RedTeamCritic",
    )

    # Structural prior (GraphSpec) knobs.
    relevance_threshold_gamma: float = 0.4
    temperature_base: float = 0.35  # β(b) baseline (budget-conditioned temperature)

    # Orchestration (token-aware) knobs for the reference demo.
    max_steps: int = 8
    group_size: int = 6  # GRPO-style group sampling in demo
    clip_eps: float = 0.2
    kl_alpha: float = 0.7  # λ in Eq. (12) style penalty weight for prior anchoring
    token_cost_beta: float = 0.02  # β in r_t = R_acc - β C_token

    # Lightweight embedding dimension for reference implementation (no external encoders).
    embed_dim: int = 96

