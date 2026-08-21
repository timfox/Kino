"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.llms_graphs.agents import agents_demo
from ltx_trainer.llms_graphs.benchmarks import PAPER_ANCHORS, benchmarks_bundle
from ltx_trainer.llms_graphs.config import LLMsGraphsConfig, TUTORIAL_SECTIONS
from ltx_trainer.llms_graphs.graph_rag import graph_rag_demo
from ltx_trainer.llms_graphs.integration import integration_bundle
from ltx_trainer.llms_graphs.kg_grounding import kg_grounding_demo
from ltx_trainer.llms_graphs.synergies import synergies_demo


def framework_card(cfg: LLMsGraphsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LLMsGraphsConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "scope": "PAKDD tutorial — unified LLM + graph + KG + agent synergies",
            "axes": list(benchmarks_bundle()["synergy_catalog"].keys()),
            "duration_minutes": sum(m for _, m in TUTORIAL_SECTIONS),
            "sections": len(TUTORIAL_SECTIONS),
            "distinct_from_prior": "End-to-end bidirectional treatment vs narrow LLM-GNN or RAG-only tutorials",
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    syn = synergies_demo()
    return {
        "synergies": syn,
        "graph_rag": graph_rag_demo(seed=seed),
        "kg_grounding": kg_grounding_demo(),
        "agents": agents_demo(),
        "ref_synergy_axes": PAPER_ANCHORS["synergy_axes"],
        "ref_tutorial_minutes": PAPER_ANCHORS["tutorial_minutes"],
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "package": "llms_graphs",
        "paper": "llms_graphs",
        "arxiv": "2606.11560",
        "ref_synergy_axes": demo["ref_synergy_axes"],
        "ref_tutorial_minutes": demo["ref_tutorial_minutes"],
        "anchor_systems": PAPER_ANCHORS["anchor_systems"],
        "graph_rag_retrieved": demo["graph_rag"]["retrieved"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
