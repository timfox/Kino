"""Framework card and benchmarks for PDE survey."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pde_survey.config import PdeSurveyConfig
from ltx_trainer.pde_survey.defenses import defense_catalog
from ltx_trainer.pde_survey.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.pde_survey.mock import evaluation_smoke
from ltx_trainer.pde_survey.tables import headline_results, table1_bundle, table2_sota_availability
from ltx_trainer.pde_survey.taxonomy import table1_scenarios


def framework_card(cfg: PdeSurveyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PdeSurveyConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Pretraining Data Exposure (PDE): determine whether data appeared in an LLM's "
            "pretraining corpus — unifying membership inference (instance-level) and "
            "data contamination (dataset-level)."
        ),
        "formalism": {
            "instance": "f(M,x) binary via matching function b(x,x')",
            "dataset_partial": "∃x∈D: f(M,x)=1",
            "dataset_full": "∀x∈D: f(M,x)=1",
            "score": "PDE(D,M) = (1/|D|) Σ_x f(M,x)",
        },
        "threat_model": {
            "access": "query-only adversary",
            "no_pretrain_corpus": True,
            "focus": "English text LLMs (open + API)",
        },
        "taxonomy": {
            "dimensions": ["application_scenario", "user_type", "security_risk"],
            "scenarios": list(cfg.scenarios),
        },
        "defense_families": [d["category"] for d in defense_catalog()],
        "domains": list(cfg.domains),
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_scenarios": table1_scenarios(),
        "table1": table1_bundle(),
        "table2_sota_availability": table2_sota_availability(),
        "defense_catalog": defense_catalog(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
