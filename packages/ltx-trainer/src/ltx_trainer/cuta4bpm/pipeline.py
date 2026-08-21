"""Framework card and benchmark bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cuta4bpm.config import Cuta4BpmConfig
from ltx_trainer.cuta4bpm.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.cuta4bpm.mock import evaluation_smoke, example_prescription_process
from ltx_trainer.cuta4bpm.tables import (
    block_to_bpmn_mapping,
    evaluation_summary,
    headline_results,
    mde_abstraction_layers,
    participative_lifecycle,
    table1_control_blocks,
)
from ltx_trainer.cuta4bpm.transform import transform_process


def framework_card(cfg: Cuta4BpmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Cuta4BpmConfig()
    return {
        "name": cfg.paper_title,
        "paper_id": cfg.paper_id,
        "authors": cfg.authors,
        "problem": (
            "Bridge domain-expert-friendly block models and technical graph-oriented "
            "BPMN for automation without invalid unstructured control flow."
        ),
        "cuta4bpm": {
            "element": "SimpleActivity card (subject–predicate–object sentence + role, docs, location)",
            "control_blocks": list(cfg.control_blocks),
            "elicitation": f"{cfg.interview_sessions} participative card sessions",
        },
        "mde_layers": mde_abstraction_layers(),
        "transformation": {
            "language": cfg.transformation_language,
            "strategy": cfg.flattening_strategy,
            "target_pim": cfg.target_pim,
            "target_psm": cfg.target_psm,
        },
        "lifecycle": participative_lifecycle(),
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    proc = example_prescription_process()
    return {
        "table1_control_blocks": table1_control_blocks(),
        "mde_layers": mde_abstraction_layers(),
        "block_to_bpmn_mapping": block_to_bpmn_mapping(),
        "participative_lifecycle": participative_lifecycle(),
        "evaluation": evaluation_summary(),
        "headlines": headline_results(),
        "example_process": proc.to_dict(),
        "example_bpmn_summary": transform_process(proc).summary(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headlines": headline_results()}
