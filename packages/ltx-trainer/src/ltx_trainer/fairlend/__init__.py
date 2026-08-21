"""FairLend — HMDA fair binning + association rules + DIR audit (arXiv:2606.12435)."""

from ltx_trainer.fairlend.audit_pipeline import run_fair_lending_audit
from ltx_trainer.fairlend.benchmarks import benchmarks_bundle
from ltx_trainer.fairlend.clustering import cluster_profiles, select_k
from ltx_trainer.fairlend.config import (
    BENCHMARK,
    FairLendConfig,
    PAPER_ARXIV,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.fairlend.dir_audit import dir_audit_summary, disparate_impact_ratio
from ltx_trainer.fairlend.fpgrowth import compare_binning_regimes, mine_denial_rules, paper_denial_rules
from ltx_trainer.fairlend.integration import framework_card, integration_bundle
from ltx_trainer.fairlend.pipeline import evaluation_demo, evaluation_smoke
from ltx_trainer.fairlend.preprocessing import cleaning_pipeline_summary
from ltx_trainer.fairlend.transactions import build_transaction, synthetic_chicago_applications

__all__ = [
    "BENCHMARK",
    "FairLendConfig",
    "PAPER_ARXIV",
    "PAPER_REPO",
    "PAPER_TITLE",
    "PAPER_URL",
    "benchmarks_bundle",
    "build_transaction",
    "cleaning_pipeline_summary",
    "cluster_profiles",
    "compare_binning_regimes",
    "dir_audit_summary",
    "disparate_impact_ratio",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "integration_bundle",
    "mine_denial_rules",
    "paper_denial_rules",
    "run_fair_lending_audit",
    "select_k",
    "synthetic_chicago_applications",
]
