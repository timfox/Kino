"""Reference tables from the paper (blocks, MDE layers, mappings)."""

from __future__ import annotations

from typing import Any


def table1_control_blocks() -> list[dict[str, str]]:
    return [
        {
            "block": "Sequence",
            "description": "Activities or nested blocks executed one after another",
        },
        {
            "block": "Case",
            "description": "Execute one out of many alternatives",
        },
        {
            "block": "Loop",
            "description": "Repeat a sequence of elements",
        },
        {
            "block": "Parallel",
            "description": "Execute at least two elements in parallel",
        },
        {
            "block": "MultipleChoice",
            "description": "One or more out of many elements executed",
        },
    ]


def mde_abstraction_layers() -> dict[str, str]:
    return {
        "CIM": "CUTA4BPM — participative analysis with domain experts (cards / editor)",
        "PIM": "BPMN 2.0 — detailed process design, platform-independent",
        "PSM": "BPEL — executable / platform-specific automation",
        "transform_CIM_PIM": "QVT operational: CUTA4BPM → BPMN (flattening subgraphs)",
    }


def block_to_bpmn_mapping() -> dict[str, str]:
    return {
        "Workflow": "BPMN process: start event, transformed body, end event + pool",
        "SimpleActivity": "BPMN task; lane per role; IO documents as data associations (paper)",
        "Sequence": "Chain of subgraphs with sequence flows",
        "Case": "Exclusive fork + join gateways; conditions on outgoing flows",
        "Parallel": "Parallel fork + join gateways",
        "MultipleChoice": "Inclusive fork + join gateways",
        "Loop_begin": "Exclusive gateway; condition on entry; back-edge to fork",
        "Loop_end": "Body then exclusive join; condition on exit to loop",
    }


def participative_lifecycle() -> list[str]:
    return [
        "Session 1: overview with CUTA cards (analyst + domain expert)",
        "Session 2: refine / rearrange cards → formal CUTA4BPM model",
        "Future workshop: as-is critique → to-be ideas",
        "Automatic transform → BPMN for technical detail and automation path",
    ]


def evaluation_summary() -> dict[str, Any]:
    return {
        "interviews_approx": 20,
        "domains": ["farming", "pharmacy", "university administration", "production"],
        "benefits": [
            "Easier structuring for non-IT experts",
            "Example-driven modeling",
            "Improved communication",
            "Reduced imprecise / incomplete models",
        ],
        "prototype_status": "CUTA4BPM graphical editor + BPMN transform implemented (per paper)",
        "planned": "Backward engineering BPMN/BPEL → CUTA4BPM",
    }


def headline_results() -> dict[str, Any]:
    return {
        "approach": "Participative forward engineering with MDE",
        "user_language": "Block-oriented CUTA4BPM (structured, deadlock-safe)",
        "technical_language": "Graph-oriented BPMN → BPEL",
        "transform": "Flattening nested blocks to BPMN subgraphs (Mendling et al.)",
    }
