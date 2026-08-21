"""GOPEX integration hooks for TrajGenAgent."""

from __future__ import annotations

from typing import Any

from ltx_trainer.trajgenagent.benchmarks import PAPER_ANCHORS, TABLE_II_NUMOSIM
from ltx_trainer.trajgenagent.config import (
    BENCHMARK_MOBILITYSYN,
    BENCHMARK_NUMOSIM,
    TrajGenAgentConfig,
    PAPER_ARXIV,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
)


def integration_metadata() -> dict[str, str]:
    return {
        "package": "ltx_trainer.trajgenagent",
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "benchmarks": f"{BENCHMARK_NUMOSIM}; {BENCHMARK_MOBILITYSYN}",
        "upstream_repo": PAPER_REPO,
        "role": "Hierarchical LLM agent for visit-wise mobility trajectory generation",
    }


def gopex_stub_links() -> dict[str, str]:
    return {
        "geo_llama": "Trajectory-level fine-tuned LLM baseline (MDM 2025)",
        "livebrowsecomp": "Agentic tool orchestration patterns",
        "physics_steering": "Constraint-guided generation priors",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "mobility_trajectory_generation",
        "pipeline": [
            "numosim_or_mobilitysyn_ingest",
            "historical_evidence_profile",
            "orchestrator_activity_chain_icl",
            "langgraph_worker_grounding",
            "spatial_poi_retrieval",
            "kinematics_travel_time",
            "llm_duration_module",
            "icad_bestad_anomaly_eval",
        ],
        "backbone": "Qwen2.5-32B-Instruct via vLLM (zero-shot)",
        "companion_impl": PAPER_REPO,
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Convert stay-point trajectories: activity, POI, ts, te per visit.",
        "Build Πu evidence: activity/transition/POI/duration/speed priors per individual.",
        "Stage 1: ICL exemplar chains + summary stats; verify vocabulary/no-adjacent-dup.",
        "Stage 2: fixed order location → travel → duration with verifier fallbacks.",
        "Peer-augment POI pool via top-K similar mobility signatures.",
        "Evaluate JSD/Frobenius stats + ICAD/BeSTAD balanced AUROC/AP (→ 0.5).",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
        "anchors": PAPER_ANCHORS,
        "table_ii_numosim_trajgenagent": next(r for r in TABLE_II_NUMOSIM if r["model"] == "TrajGenAgent"),
    }


def framework_card(cfg: TrajGenAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrajGenAgentConfig()
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "method": {
            "architecture": "orchestrator–worker hierarchical agent",
            "stage1": "LLM activity-chain ICL (macro semantic scaffold)",
            "stage2": "deterministic LangGraph worker (micro spatiotemporal grounding)",
            "backbone": cfg.backbone_llm,
            "fine_tuning": False,
        },
        "config": cfg.__dict__,
        "integration": integration_metadata(),
    }
