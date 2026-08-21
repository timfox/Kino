"""DeepSWE configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.deepswe.constants import (
    CANARY_GUID,
    GITHUB_REPO,
    HARNESS_NAME,
    N_REPOS,
    N_TASKS,
    WEBSITE,
)


@dataclass
class DeepSWEConfig:
    name: str = "DeepSWE"
    title: str = "Measuring frontier coding agents on original, long-horizon engineering tasks"
    authors: str = "Wenqi Huang, Charley Lee, Leonard Tng, Serena Ge"
    published: str = "2026-05-26"
    paper_arxiv: str = ""
    website: str = WEBSITE
    github_repo: str = GITHUB_REPO
    harness: str = HARNESS_NAME
    n_tasks: int = N_TASKS
    n_repos: int = N_REPOS
    canary_guid: str = CANARY_GUID
    ltx_hook: str = (
        "Agent harness QA for GOPEX stack changes via mini-swe-agent-style bash loops; "
        "behavioral verifiers for Pier/n8n automation PRs"
    )
    citation_bibtex: str = field(
        default_factory=lambda: """@misc{datacurve2026deepswe,
  title  = {DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks},
  author = {Wenqi Huang and Charley Lee and Leonard Tng and Serena Ge},
  year   = {2026},
  url    = {https://github.com/datacurve-ai/deep-swe},
}"""
    )
