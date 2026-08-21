"""Framework card and benchmarks for SpecX."""

from __future__ import annotations

from typing import Any

from ltx_trainer.specx.config import SpecxConfig
from ltx_trainer.specx.layout import (
    BENCHMARK_COMPARISON,
    LIMITATIONS,
    TASK_DESCRIPTIONS,
    TIER_DESCRIPTIONS,
)
from ltx_trainer.specx.mock import evaluation_smoke, pipeline_demo
from ltx_trainer.specx.tables import (
    modality_representations,
    table1_benchmark_comparison,
    table3_elucidation_random_excerpt,
    table3_elucidation_scaffold_excerpt,
    table4_functional_group_random,
    table6_qa_smiles_small_excerpt,
    table11_subset_modalities,
)


def framework_card(cfg: SpecxConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpecxConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "institution": cfg.institution,
        "summary": (
            "1.7M-molecule multimodal spectroscopy benchmark with cross-paradigm "
            "evaluation of specialized ML models vs MLLMs."
        ),
        "scale": {
            "filtered_molecules": cfg.total_molecules_filtered,
            "retention_pct": cfg.retention_rate_pct,
        },
        "modalities": list(cfg.modalities_all),
        "tiers": {
            k: {
                "molecules": getattr(cfg, f"tier_{k.lower()}_molecules"),
                "description": v,
            }
            for k, v in TIER_DESCRIPTIONS.items()
        },
        "tasks": {k: TASK_DESCRIPTIONS[k] for k in cfg.tasks_ml + cfg.tasks_qa},
        "splits": list(cfg.split_strategies),
        "data_sources": list(cfg.sources),
        "findings": [
            "NMR (1H, 13C) best single-modality for elucidation; multimodal NMR+MS reaches 59% Top-1 (random).",
            "Raman/IR strongest for functional-group macro-F1 (XGBoost up to 0.97 scaffold).",
            "MLLM QA Top-1 near zero on SMILES inference; specialized models dominate signal-level tasks.",
            "Scaffold splits show large OOD generalization gap vs random splits.",
        ],
        "benchmark_vs_prior": BENCHMARK_COMPARISON,
        "limitations": list(LIMITATIONS),
        "mllm_eval": list(cfg.mllm_eval_models),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_comparison": table1_benchmark_comparison(),
        "table3_elucidation_random": table3_elucidation_random_excerpt(),
        "table3_elucidation_scaffold": table3_elucidation_scaffold_excerpt(),
        "table4_functional_group": table4_functional_group_random(),
        "table6_qa_smiles": table6_qa_smiles_small_excerpt(),
        "table11_subsets": table11_subset_modalities(),
        "modality_representations": modality_representations(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "pipeline": pipeline_demo()}


def headline_results() -> dict[str, Any]:
    return {
        "best_elucidation_top1_random": 59.04,
        "best_elucidation_modality_combo": "13C-NMR + 1H-NMR + MS",
        "best_fg_f1_raman_xgb": 0.958,
        "mllm_qa_top1_best": 0.015,
    }
