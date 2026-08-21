"""Six-phase orchestration methodology (Sec. 3.3)."""

from __future__ import annotations

from typing import Any


def phase_catalog() -> list[dict[str, Any]]:
    return [
        {
            "phase": 1,
            "name": "domain_inventory",
            "output": "domain inventory with work products and current production method",
            "complete_when": "every deliverable type mapped to a domain",
        },
        {
            "phase": 2,
            "name": "tool_mapping",
            "output": "tool-domain mapping table with rationale or unmappable flag",
            "complete_when": "each domain mapped or documented unmappable",
        },
        {
            "phase": 3,
            "name": "skill_transfer_assessment",
            "output": "skill transfer matrix (input, validation, iteration, integration)",
            "complete_when": "all four dimensions assessed per tool-domain pair",
        },
        {
            "phase": 4,
            "name": "integration_design",
            "output": "cross-domain workflow specs with gates and governance checkpoints",
            "complete_when": "each multi-tool workflow reviewed for execution",
        },
        {
            "phase": 5,
            "name": "orchestration_execution",
            "output": "work products + portability metrics + failure log",
            "complete_when": "one full cycle per workflow with metrics collected",
        },
        {
            "phase": 6,
            "name": "portfolio_optimization",
            "output": "revised mappings, designs, metrics",
            "complete_when": "failure cases reviewed with optimization rationale (recurring)",
        },
    ]


def orchestration_patterns() -> list[dict[str, str]]:
    """Sec. 3.5 — six inductive patterns from case study."""
    return [
        {"id": 1, "name": "authority_document_chain"},
        {"id": 2, "name": "format_translation"},
        {"id": 3, "name": "quality_gate_handoff"},
        {"id": 4, "name": "governance_checkpoint"},
        {"id": 5, "name": "terminal_node_placement"},
        {"id": 6, "name": "automated_authority_transfer"},
    ]
