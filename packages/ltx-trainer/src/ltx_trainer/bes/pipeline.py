"""BES framework and knowledge cards."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bes.benchmarks import benchmarks_bundle
from ltx_trainer.bes.config import BESConfig
from ltx_trainer.bes.constants import GITHUB_REPO, OPERATOR_PROBS
from ltx_trainer.bes.evaluation import evaluation_demo, evaluation_smoke
from ltx_trainer.bes.experiments import experiments_bundle
from ltx_trainer.bes.theory import theory_card


def framework_card(cfg: BESConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BESConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "github": cfg.github,
        "authors": "Guowei Xu, Zhenting Qi, Huangyuan Su, Weirui Ye, "
        "Himabindu Lakkaraju, Sham M. Kakade, Yilun Du",
        "affiliations": ["Harvard University", "MIT"],
        "forward_operators": list(OPERATOR_PROBS.keys()),
        "backward": "recursive sub-goal decomposition with dense verifiers",
        "limitations": [
            "requires objective or judge signal",
            "backward quality depends on decomposer model",
            "post-training tested up to 8B scale",
        ],
        "ltx_hook": "Replace best-of-N / tree rollouts in GOPEX agent and caption loops",
        "upstream_code": GITHUB_REPO,
    }


def knowledge_card(cfg: BESConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BESConfig()
    return {
        "framework": framework_card(cfg),
        "benchmarks": benchmarks_bundle(),
        "experiments": experiments_bundle(),
        "theory": theory_card(),
    }


__all__ = [
    "framework_card",
    "knowledge_card",
    "evaluation_demo",
    "evaluation_smoke",
]
