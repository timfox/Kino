"""Participative lifecycle stages and stub limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "Participative elicitation: CUTA card interviews (2 sessions, as-is / to-be)",
    "CIM: structured CUTA4BPM block model with domain experts",
    "MDE forward transform: CUTA4BPM → BPMN (QVT, flattening subgraphs)",
    "PIM: detailed BPMN design and extension for automation",
    "PSM: BPEL (or platform-specific) for execution",
    "Optional backward engineering: BPMN/BPEL → CUTA4BPM (planned in paper)",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — no QVT engine, graphical CUTA editor, or BPMN XML export.",
    "Toy in-memory graph (nodes/edges) approximates BPMN subgraph patterns only.",
    "Does not validate full BPMN 2.0 conformance or generate BPEL.",
)
