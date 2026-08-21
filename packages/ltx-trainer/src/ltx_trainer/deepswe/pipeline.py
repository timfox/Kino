"""DeepSWE framework card and knowledge exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.benchmarks import benchmarks_bundle
from ltx_trainer.deepswe.config import DeepSWEConfig
from ltx_trainer.deepswe.core import separation_vs_swe_bench_pro
from ltx_trainer.deepswe.evaluation import evaluation_demo
from ltx_trainer.deepswe.harness import harness_card
from ltx_trainer.deepswe.methodology import methodology_card
from ltx_trainer.deepswe.qualitative import qualitative_bundle
from ltx_trainer.deepswe.run_plan import run_plan


def framework_card(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeepSWEConfig()
    return {
        "name": cfg.name,
        "title": cfg.title,
        "authors": cfg.authors,
        "published": cfg.published,
        "website": cfg.website,
        "github_repo": cfg.github_repo,
        "harness": cfg.harness,
        "n_tasks": cfg.n_tasks,
        "n_repos": cfg.n_repos,
        "canary_guid": cfg.canary_guid,
        "ltx_hook": cfg.ltx_hook,
        "advances": [
            "contamination_free",
            "high_diversity_91_repos",
            "long_horizon_short_prompts",
            "behavioral_verifiers",
        ],
        "methodology": methodology_card(),
        "harness": harness_card(),
        "separation": separation_vs_swe_bench_pro(),
        "benchmarks": benchmarks_bundle(),
    }


def knowledge_card(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeepSWEConfig()
    return {
        "framework": framework_card(cfg),
        "qualitative": qualitative_bundle(),
        "run_plan": run_plan(cfg),
        "citation": cfg.citation_bibtex,
    }


# Re-export for backward compatibility
def evaluation_smoke(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.deepswe.evaluation import evaluation_smoke as _smoke

    return _smoke(cfg)
