"""Framework card and benchmark bundles for GEM."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gem.config import GemConfig
from ltx_trainer.gem.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.gem.mock import evaluation_smoke
from ltx_trainer.gem.tables import (
    ablation_clustering_avg,
    headline_results,
    k_sensitivity_perf,
    student_distillation_accuracy,
    table1_gem_breakdown,
    table1_main_average,
)


def framework_card(cfg: GemConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GemConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "author": cfg.author,
        "problem": (
            "LLM pre-training mixing needs semantically balanced partitions on the "
            "hypersphere; Euclidean k-means suffers anisotropic cluster collapse."
        ),
        "method": {
            "likelihood": "von Mises–Fisher mixture on S^{d-1}",
            "prior": "Fixed α_k = 1/K (decoupled from empirical mass)",
            "regularizer": "R(π) = −λ/2 ‖π − u‖² on empirical cluster mass π(Γ)",
            "inference": "MM E-step with guaranteed surrogate ascent + closed-form M-step",
            "scalability": "Teacher–student FastText distillation + GIS taxonomy labels",
        },
        "mixing_integrations": ["DoReMi", "Perf", "RegMix"],
        "default_k": cfg.n_clusters,
        "balance_lambda": cfg.balance_lambda,
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_main_average": table1_main_average(),
        "table1_gem_breakdown": table1_gem_breakdown(),
        "ablation_clustering": ablation_clustering_avg(),
        "k_sensitivity_perf": k_sensitivity_perf(),
        "student_distillation": student_distillation_accuracy(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headlines": headline_results()}
