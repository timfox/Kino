"""Example hospital-adjacent process + transform smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cuta4bpm.config import Cuta4BpmConfig
from ltx_trainer.cuta4bpm.metamodel import Block, CutaProcess, Role, SimpleActivity
from ltx_trainer.cuta4bpm.transform import transform_process
from ltx_trainer.cuta4bpm.validate import block_depth, count_activities, is_structured_element


def example_prescription_process() -> CutaProcess:
    """Toy process inspired by pharmacy / hospital administration domains in the paper."""
    return CutaProcess(
        name="Medication order handling",
        company_pool="University Hospital",
        roles=[
            Role("Physician", "Clinical ward"),
            Role("Pharmacist", "Pharmacy unit"),
        ],
        root=Block(
            kind="Sequence",
            children=[
                SimpleActivity(
                    subject="Physician",
                    predicate="prescribes",
                    obj="medication for patient",
                    role="Physician",
                    seq_no=1,
                    output_documents=["prescription"],
                ),
                Block(
                    kind="Case",
                    conditions=["urgent", "routine"],
                    children=[
                        SimpleActivity(
                            subject="Pharmacist",
                            predicate="dispenses",
                            obj="urgent medication",
                            role="Pharmacist",
                            input_documents=["prescription"],
                            seq_no=2,
                        ),
                        Block(
                            kind="Parallel",
                            children=[
                                SimpleActivity(
                                    subject="Pharmacist",
                                    predicate="verifies",
                                    obj="prescription",
                                    role="Pharmacist",
                                    seq_no=3,
                                ),
                                SimpleActivity(
                                    subject="Nurse",
                                    predicate="prepares",
                                    obj="patient record",
                                    role="Clinical ward",
                                    seq_no=4,
                                ),
                            ],
                        ),
                    ],
                ),
                SimpleActivity(
                    subject="Nurse",
                    predicate="administers",
                    obj="medication to patient",
                    role="Clinical ward",
                    input_documents=["prescription"],
                    seq_no=5,
                ),
            ],
        ),
    )


def evaluation_smoke(cfg: Cuta4BpmConfig | None = None) -> dict[str, Any]:
    c = cfg or Cuta4BpmConfig()
    proc = example_prescription_process()
    bpmn = transform_process(proc)
    root = proc.root
    return {
        "paper": c.paper_id,
        "process_name": proc.name,
        "structured": is_structured_element(root),
        "activity_count": count_activities(root),
        "nesting_depth": block_depth(root),
        "bpmn_summary": bpmn.summary(),
        "has_parallel_gateway": any(n.kind == "parallel_gateway" for n in bpmn.nodes),
        "has_exclusive_gateway": any(n.kind == "exclusive_gateway" for n in bpmn.nodes),
        "task_count": sum(1 for n in bpmn.nodes if n.kind == "task"),
    }
