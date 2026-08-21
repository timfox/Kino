"""GOPEX integration hooks for FairLend."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fairlend.benchmarks import PAPER_ANCHORS, TABLE_3_BINNING
from ltx_trainer.fairlend.config import BENCHMARK, FairLendConfig, PAPER_ARXIV, PAPER_REPO, PAPER_TITLE, PAPER_URL


def integration_metadata() -> dict[str, str]:
    return {
        "package": "ltx_trainer.fairlend",
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "benchmark": BENCHMARK,
        "upstream_repo": PAPER_REPO,
        "role": "HMDA fair-binning + FP-Growth + cluster DIR lending audit",
    }


def gopex_stub_links() -> dict[str, str]:
    return {
        "ethical_pluralism": "Normative fairness ensembles for policy review",
        "constraint_tax": "Structured-output validity on audit reports",
        "gem": "Hyperspherical mixing for stratified HMDA sampling",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "fair_lending_audit",
        "pipeline": [
            "hmda_chicago_ingest",
            "pyspark_clean_bin",
            "epsilon_fair_binning",
            "fpgrowth_denial_rules",
            "kmeans_financial_clusters",
            "dir_disparate_impact_audit",
        ],
        "data_source": "CFPB HMDA 2023 Snapshot (MSA 16984)",
        "companion_impl": PAPER_REPO,
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Extract HMDA CSV from zip; filter derived_msa_md=16984 (Chicago division).",
        "Map NA/Exempt strings; clip income 0.5–99.5 pct; DTI range → midpoint.",
        "Run equal-frequency vs ε-biased D&C binning on income and loan_amount.",
        "Build 11-item transactions; FP-Growth min_support=0.10, conf≥0.50, lift≥1.0.",
        "K-Means on z-scored income, loan, DTI, CLTV (exclude interest_rate).",
        "DIR audit per cluster with min group size 30; flag DIR < 0.80.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
        "anchors": PAPER_ANCHORS,
        "table_3_binning": TABLE_3_BINNING,
    }


def framework_card(cfg: FairLendConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FairLendConfig()
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "method": {
            "stages": ["clean+bin", "FP-Growth", "K-Means+DIR"],
            "fair_binning_ref": "Asudeh et al. ε-biased D&C (arXiv:2509.21785)",
            "benchmark": BENCHMARK,
        },
        "config": cfg.__dict__,
        "integration": integration_metadata(),
    }
