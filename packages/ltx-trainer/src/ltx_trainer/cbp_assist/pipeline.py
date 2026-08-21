"""Hybrid capability-based SMT planning assistance API (arXiv:2605.28666)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cbp_assist.benchmarks import benchmarks_bundle
from ltx_trainer.cbp_assist.config import AGENT_ROLES, COMPONENTS, CbpAssistConfig
from ltx_trainer.cbp_assist.examples import (
    demo_adaptive_planning,
    demo_knowledge_query,
    demo_runtime_failure,
    demo_sat_planning,
    demo_unsat_planning,
)


def framework_card(cfg: CbpAssistConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CbpAssistConfig()
    return {
        "name": "Hybrid CBP-LLM Assistance",
        "paper": f"arXiv:{cfg.arxiv}",
        "institution": cfg.institution,
        "components": list(COMPONENTS),
        "agents": list(AGENT_ROLES),
        "orchestration": cfg.orchestration,
        "symbolic_backend": "SMT capability planning [Köcher et al. CAIPI 2024]",
        "llm_layer": "NL grounding, explanation, adaptation (HitL-gated)",
        "evaluation": {
            "platform": "MPS500 modular production system",
            "test_cases": cfg.test_cases_total,
            "table1": cfg.table1_successful,
        },
        "llm_models": {"default": cfg.llm_default, "analyzer": cfg.llm_analyzer},
        "data_repo": cfg.mps500_repo,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28666",
        "authors": "Vieira da Silva, König, Gehlhoff (HSU Hamburg)",
        "title": "LLM-Based Assistance for Capability-Based SMT Planning",
        "architecture": "Formal planner for correctness; LLM for NL access, UNSAT explanation, adaptation",
        "workflow": "Routed LangGraph: Router → 5 specialists; 3 HitL checkpoints",
        "results_table1": {
            "knowledge_query": "9/10",
            "sat_planning": "4/4",
            "unsat_planning": "3/4",
            "adaptive_planning": "5/5",
        },
        "finding": "Hybrid design improves accessibility without sacrificing symbolic planning guarantees",
    }


def evaluation_demo() -> dict[str, Any]:
    kq = demo_knowledge_query()
    sat = demo_sat_planning()
    unsat = demo_unsat_planning()
    adapt = demo_adaptive_planning()
    runtime = demo_runtime_failure()
    return {
        "knowledge_query": kq,
        "sat_planning": sat,
        "unsat_planning": unsat,
        "adaptive_planning": adapt,
        "runtime_failure": runtime,
        "all_demos_ok": all(
            d["ok"] for d in (kq, sat, unsat, adapt, runtime)
        ),
    }


def evaluation_smoke(cfg: CbpAssistConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo()
    bench = benchmarks_bundle()
    t1 = bench["table1"]
    return {
        "paper": "arXiv:2605.28666",
        "all_demos_ok": demo["all_demos_ok"],
        "test_cases": bench["total_test_cases"],
        "table1_kq": next(r for r in t1 if r["scenario"] == "Knowledge Query")["successful"],
        "ok": demo["all_demos_ok"] and bench["total_test_cases"] == 23,
    }
