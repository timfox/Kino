"""Configuration for CUTA4BPM → BPMN MDE stub."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cuta4BpmConfig:
    paper_id: str = "ICIST-2012-CUTA4BPM"
    paper_title: str = (
        "From user-understandable to technical process model: "
        "a model-driven approach using CUTA4BPM"
    )
    authors: str = (
        "Kathrin Kirchner (University Hospital Jena); "
        "Siniša Nešković, Dejan Stojimirović (University of Belgrade)"
    )
    transformation_language: str = "QVT operational (OMG MOF 2.0 QVT 1.1)"
    target_pim: str = "BPMN 2.0"
    target_psm: str = "BPEL"
    flattening_strategy: str = "Mendling et al. flattening (recursive nested blocks → subgraphs)"

    control_blocks: tuple[str, ...] = (
        "Sequence",
        "Case",
        "Loop",
        "Parallel",
        "MultipleChoice",
    )
    interview_sessions: int = 2
    evaluation_domains: tuple[str, ...] = field(
        default_factory=lambda: (
            "farming",
            "pharmacy",
            "university administration",
            "production",
        )
    )
    expert_interviews_approx: int = 20
