"""Paper table excerpts (Augment Engineering, arXiv:2605.26146)."""

from __future__ import annotations

from typing import Any


def table3_stack_inventory() -> list[dict[str, Any]]:
    """Table 3 — orchestration stack (abbreviated)."""
    return [
        {"tool": "Claude", "category": "AI tool", "transfer_note": "Primary baseline"},
        {"tool": "Gamma.app", "category": "AI tool", "transfer_velocity_h": 2},
        {"tool": "HeyGen", "category": "AI tool", "transfer_velocity_h": 4},
        {"tool": "Jira", "category": "infrastructure", "transfer_note": "Traditional SWE, not PE/CE transfer"},
    ]


def table6_pipeline_overhead() -> list[dict[str, Any]]:
    """Table 6 — Oh estimates (%)."""
    return [
        {"workflow": "academic_paper", "oh_pct": 12, "cb": 2},
        {"workflow": "training_content", "oh_pct": 18, "cb": 3},
        {"workflow": "product_development", "oh_pct": 8, "cb": 2},
        {"workflow": "contract_deliverable", "oh_pct": 5, "cb": 1},
    ]


def longitudinal_phases() -> list[dict[str, Any]]:
    """Figure 4 phase metrics (paper-reported)."""
    return [
        {
            "phase": 1,
            "period": "Nov–Dec 2025",
            "interactions": "1–13",
            "context_files_avg": 0.8,
            "first_pass_acceptance_pct": 15,
            "handoffs_per_project": 0,
            "coverage_breadth": 3,
        },
        {
            "phase": 2,
            "period": "Jan 2026",
            "interactions": "14–41",
            "context_files_avg": 2.4,
            "first_pass_acceptance_pct": 30,
            "handoffs_per_project": 0.5,
            "coverage_breadth": 5,
        },
        {
            "phase": 3,
            "period": "Feb–Mar 2026",
            "interactions": "42–102",
            "context_files_avg": 5.2,
            "first_pass_acceptance_pct": 42,
            "handoffs_per_project": 2.1,
            "coverage_breadth": 7,
        },
    ]


def headline_results() -> dict[str, Any]:
    return {
        "three_discipline_progression": "Prompt Engineering → Context Engineering → Augment Engineering",
        "cochran_armitage_z": 3.04,
        "cochran_armitage_p": "<0.01",
        "wright_law_beta": 0.44,
        "wright_law_n_artifacts": 82,
        "case_study_cb": 7,
        "production_code_loc": 41392,
        "test_pass_rate_pct": 100,
    }
