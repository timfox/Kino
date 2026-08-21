"""Paper Table 1–2 excerpts."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pde_survey.taxonomy import table1_scenarios


def table2_sota_availability() -> list[dict[str, Any]]:
    """Table 2 — code/dataset/benchmark availability (paper flags)."""
    return [
        {"ref": "2", "level": "dataset", "type": "detection", "code": False, "dataset": False, "benchmark": True},
        {"ref": "24", "level": "instance", "type": "detection", "code": True, "dataset": True, "benchmark": True},
        {"ref": "41", "level": "dataset", "type": "detection", "code": True, "dataset": True, "benchmark": True},
        {"ref": "48", "level": "instance", "type": "detection", "code": True, "dataset": True, "benchmark": True},
        {"ref": "11", "level": "dataset", "type": "detection", "code": False, "dataset": True, "benchmark": True},
        {"ref": "35", "level": "instance", "type": "detection", "code": True, "dataset": True, "benchmark": False},
        {"ref": "28", "level": "dataset", "type": "mitigation", "code": True, "dataset": True, "benchmark": True},
        {"ref": "59", "level": "dataset", "type": "mitigation", "code": True, "dataset": True, "benchmark": True},
        {"ref": "14", "level": "instance", "type": "mitigation", "code": False, "dataset": True, "benchmark": True},
    ]


def headline_results() -> dict[str, Any]:
    return {
        "unified_framework": "Instance-level PDE (MIA) + dataset-level PDE (contamination)",
        "exposure_score": "PDE(D,M) ∈ [0,1] partial exposure between binary extremes",
        "threat_model": "Query access; no pretraining data; English text LLMs",
        "future_directions": [
            "targeted unlearning",
            "model explainability for memorization",
            "semantic-level contamination detection",
        ],
    }


def table1_bundle() -> dict[str, Any]:
    return {"scenarios": table1_scenarios()}
