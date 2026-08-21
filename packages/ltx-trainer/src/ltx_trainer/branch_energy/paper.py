"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.branch_energy.benchmarks import summary_anchors, table_i_rows
from ltx_trainer.branch_energy.branch import branch_card
from ltx_trainer.branch_energy.classical import balanced_rl_phase_comparison, classical_card, classical_comparisons
from ltx_trainer.branch_energy.constants import (
    PAPER_ARXIV,
    PAPER_CODE,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TEST_CASES,
    THEOREMS,
    TRAIN_DEFAULTS,
)
from ltx_trainer.branch_energy.theorems import theorems_card
from ltx_trainer.branch_energy.topology import topology_card
from ltx_trainer.branch_energy.references import REFERENCES


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "code": PAPER_CODE,
        "authors": "F.G. Montoya, F. de Leon, F.M. Arrabal-Campos, A. Alcayde",
        "findings": [
            "Branch-Level Localization Theorem: unique p_R, W'_L, W'_C given admissible topology",
            "Topology-Indeterminacy Theorem: same terminals, distinct branch maps",
            "Generalized Energetic Duality: Norton–Thevenin / L↔C as LTI sinusoidal limits",
            "Six test cases vs IEEE 1459, p–q, CPC, FBD (Table I)",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "theorems": theorems_card(),
        "branch_balance": branch_card(),
        "topology": topology_card(),
        "classical_theories": classical_card(),
        "iv_a_comparison": balanced_rl_phase_comparison(),
        "classical_comparisons": classical_comparisons(),
        "test_cases": list(TEST_CASES),
        "table_i": table_i_rows(),
        "theorem_statements": list(THEOREMS),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "summary": summary_anchors(),
        "references": list(REFERENCES),
    }
