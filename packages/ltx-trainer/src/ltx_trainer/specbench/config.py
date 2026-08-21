"""SpecBench configuration (arXiv:2605.30314)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpecBenchConfig:
    """RFC specification-deficiency benchmark parameters."""

    paper_arxiv: str = "2605.30314"
    github_url: str = "https://github.com/kevins981/SpecBench"
    n_repositories: int = 5
    repositories: tuple[str, ...] = ("kubernetes", "react", "rust", "tvm", "vllm")
    prediction_budget_multiplier: float = 1.25
    core_weight: float = 1.0
    extended_weight: float = 0.5
    core_likert_threshold: float = 3.0
    core_endorsement_fraction: float = 2 / 3
    judge_trials: int = 4
    judge_majority: int = 3
    judge_models: tuple[str, ...] = ("gpt-5.4", "claude-sonnet-4.6")
    spi_subject_overlap: float = 0.18
    spi_predicate_overlap: float = 0.14
    # Paper headline overall accuracy (Figure 3)
    paper_best_agent: str = "codex-5.4"
    paper_best_accuracy: float = 0.444
    agent_results: tuple[tuple[str, float], ...] = (
        ("codex-5.4", 0.444),
        ("claude-opus-4.6", 0.421),
        ("claude-sonnet-4.6", 0.418),
        ("gemini-2.5-pro", 0.392),
        ("gpt-5.2", 0.385),
    )
    random_seed: int = 42

    @classmethod
    def production(cls) -> SpecBenchConfig:
        return cls()
