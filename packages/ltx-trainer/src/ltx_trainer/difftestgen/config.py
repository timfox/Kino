"""DiffTestGen — change-directed LLM differential testing (arXiv:2607.16024)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.16024"
PAPER_TITLE = (
    "DiffTestGen: Change-Directed LLM-Based Testing for Exposing Behavioral Differences"
)
PAPER_SYSTEM = "DiffTestGen"
PAPER_AUTHORS = (
    "Huimin Hu (CISPA), Cristian Cadar (Imperial College London), "
    "Michael Pradel (CISPA)"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_GITHUB = "https://github.com/sola-st/DiffTestGen"
BENCHMARK = (
    "463 PRs (Testora 439 + ChaCo 34, −10 overlap) across keras / marshmallow / pandas / scipy"
)

COMPONENTS = (
    "change_analysis",
    "static_access_information",
    "union_coverage_feedback",
    "behavioral_difference_oracle",
)

FUNCTION_CATEGORIES = ("public", "private", "special")
BASELINES = ("Testora", "Testora++", "ChaCo")


@dataclass
class DiffTestGenConfig:
    """Runtime knobs for the CPU stub."""

    max_static_fix_attempts: int = 5
    max_runtime_fix_attempts: int = 5
    initial_tests: int = 20
    top_k_call_paths: int = 5
    max_rounds: int = 5
    target_union_coverage: float = 1.0
