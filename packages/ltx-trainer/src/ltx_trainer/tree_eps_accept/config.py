"""Tree ε-acceptance configuration (arXiv:2605.27192)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TreeEpsAcceptConfig:
    paper_arxiv: str = "2605.27192"
    paper_title: str = (
        "Tree Automata Acceptance up to Measurable Defect"
    )
    authors: str = (
        "Anita Moyasari, Harsh Beohar, Charles Grellois (Sheffield); "
        "Clemens Kupke (Strathclyde)"
    )

    sigma0_default: str = "leaf"
    sigma2_default: str = "branch"

    # Coupling-based distance lifting parameter (Sec. 3.1); Sec. 5 uses arithmetic mean
    default_lift: str = "arithmetic_mean"  # f(x,y) = (x+y)/2

    main_theorem: str = (
        "T is ε0-accepted iff ∃ T′ rigidly accepted with bd(T′, T) ≤ ε0 "
        "(deadlock-free automaton; Thm. 4.1)"
    )

    section5_examples: tuple[str, ...] = (
        "termination_leaf_mass",
        "failed_execution_error_mass",
    )
