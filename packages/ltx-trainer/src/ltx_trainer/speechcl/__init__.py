"""Representation-centric continual learning for speech and audio."""

from ltx_trainer.speechcl.config import SpeechClConfig
from ltx_trainer.speechcl.geometry import drift_report
from ltx_trainer.speechcl.lalm_stages import LalmStage, hybrid_cl_consensus, stage_transitions
from ltx_trainer.speechcl.layout import LIMITATIONS
from ltx_trainer.speechcl.mitigation import MitigationMechanism, describe_mitigation, mitigation_reference_map
from ltx_trainer.speechcl.mock import compare_adaptation_modes
from ltx_trainer.speechcl.open_problems import open_problems
from ltx_trainer.speechcl.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_findings,
    pipeline_demo,
    table_adaptation_layers,
    table_foundation_models,
    table_geometry_taxonomy,
)
from ltx_trainer.speechcl.taxonomy import (
    AdaptationLayer,
    GeometryEvolution,
    classical_vs_representation_centric,
    describe_adaptation_layer,
    describe_geometry,
)

__all__ = [
    "LIMITATIONS",
    "AdaptationLayer",
    "GeometryEvolution",
    "LalmStage",
    "MitigationMechanism",
    "SpeechClConfig",
    "benchmarks_bundle",
    "classical_vs_representation_centric",
    "compare_adaptation_modes",
    "describe_adaptation_layer",
    "describe_geometry",
    "describe_mitigation",
    "drift_report",
    "evaluation_demo",
    "framework_card",
    "headline_findings",
    "hybrid_cl_consensus",
    "mitigation_reference_map",
    "open_problems",
    "pipeline_demo",
    "stage_transitions",
    "table_adaptation_layers",
    "table_foundation_models",
    "table_geometry_taxonomy",
]
