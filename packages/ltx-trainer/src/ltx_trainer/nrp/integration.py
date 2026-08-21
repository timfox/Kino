"""GOPEX integration hooks for NRP."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nrp.benchmarks import PAPER_ANCHORS, TABLE_I
from ltx_trainer.nrp.config import NrpConfig, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def integration_metadata() -> dict[str, str]:
    return {
        "package": "ltx_trainer.nrp",
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "role": "declarative neural-relational query engine over embedded databases",
    }


def gopex_stub_links() -> dict[str, str]:
    return {
        "urng": "Interval-aware vector retrieval over graph-derived features",
        "kd_truss": "Temporal cohesive subgraph queries on relational exports",
        "llms_graphs": "Graph-native RAG over embedded tuple features",
        "gopex_hybrid_memory": "FluxMem routing for neuro-symbolic query plans",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "neuro_relational_query",
        "pipeline": [
            "embedded_database_ingest",
            "nrp_rule_compile",
            "neuro_relational_algebra_lower",
            "pytorch_transform_train",
            "gated_flat_query_eval",
        ],
        "representation": "e-facts R(c)⟨e⟩ with conj/disj/transform rules",
        "fragments": ["zero-ary", "monadic", "frontier-guarded", "ReLU-FFN+FOCQ"],
        "companion_impl": "https://arxiv.org/abs/2605.24207",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Export relational tables + embedding columns (k⟨d⟩ extended arity schema).",
        "Compile monadic rules to DHN layers; frontier-guarded via row-id expansion.",
        "Train ReLU-FFN μ parameters; gate flat queries with simple acceptance policies.",
        "Validate FOCQ equivalence on ordered Boolean structures before GPU scale-up.",
        "Pair with RelBench / Griffin graph exports for end-to-end relational ML.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
        "anchors": PAPER_ANCHORS,
        "table_i": TABLE_I,
    }


def framework_card(cfg: NrpConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NrpConfig()
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "method": {
            "rules": ["conjunction (join+⊙+sum)", "disjunction (union+sum)", "transformation (μ)"],
            "fragments": TABLE_I,
            "logic": "FOCQ / uniform TC0 on ordered Boolean DBs (ReLU-FFN)",
        },
        "config": cfg.__dict__,
        "integration": integration_metadata(),
    }
