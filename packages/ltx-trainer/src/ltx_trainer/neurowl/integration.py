"""Gopex / LTX integration metadata for NeurOWL."""

from __future__ import annotations

from typing import Any

from ltx_trainer.neurowl.config import (
    BENCHMARK,
    NeurOWLConfig,
    PAPER_ARXIV,
    PAPER_ONT_REPO,
    PAPER_TITLE,
    PAPER_URL,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.neurowl",
        "tools": [
            "neurowl_knowledge",
            "neurowl_benchmarks",
            "neurowl_eval_demo",
            "neurowl_smoke",
            "neurowl_ltx_plan",
            "neurowl_reason",
        ],
        "scripts": ["./scripts/kino-neurowl.sh", "./scripts/gopex-neurowl.sh"],
        "aiml": "gopex_agent/fixtures/neurowl.aiml",
        "doc": "documents/NEUROWL.md",
    }


def framework_card(config: NeurOWLConfig | None = None) -> dict[str, Any]:
    cfg = config or NeurOWLConfig()
    return {
        "name": "NeurOWL",
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "ont_repo": PAPER_ONT_REPO,
        },
        "benchmark": BENCHMARK,
        "stages": ["1-reasoner", "2a-downward", "2b-upward", "3a-bidirectional", "3b-direct"],
        "embedding_default": cfg.embedding_backend,
        "llm_default": cfg.llm_name,
        "top_k": cfg.top_k,
        "max_missing_axioms": cfg.max_missing_axioms,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "Incomplete OWL/EL TBox + candidate subsumption A ⊑ B",
            "stage1": "DL reasoner entailment + justification",
            "stage2": "Logical bridging via direct children/parents + LLM verify",
            "stage3": "OnT/SBERT bridge retrieval + direct LLM check",
            "export": "True/False + ≤2 missing axioms + explanation set",
            "notes": [
                "Generalized TBox abduction without predefined hypothesis space",
                "Handles incorrect target subsumptions (negative cases)",
                "OnT fine-tune matters for ∃r.B complex bridges (Snomed∃)",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub uses transitive closure + rule LLM + overlap embeddings",
            "Wire Qwen3.5-9B + ontology-transformer OnT for live runs",
        ],
    }
