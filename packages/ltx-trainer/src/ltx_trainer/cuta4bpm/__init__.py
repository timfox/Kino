"""CUTA4BPM participative MDE → BPMN stub (Kirchner et al., ICIST 2012)."""

from ltx_trainer.cuta4bpm.config import Cuta4BpmConfig
from ltx_trainer.cuta4bpm.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.cuta4bpm.metamodel import Block, CutaProcess, Role, SimpleActivity
from ltx_trainer.cuta4bpm.mock import evaluation_smoke, example_prescription_process
from ltx_trainer.cuta4bpm.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.cuta4bpm.tables import headline_results, table1_control_blocks
from ltx_trainer.cuta4bpm.transform import transform_process
from ltx_trainer.cuta4bpm.validate import block_depth, count_activities, is_structured_element

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "Block",
    "Cuta4BpmConfig",
    "CutaProcess",
    "Role",
    "SimpleActivity",
    "benchmarks_bundle",
    "block_depth",
    "count_activities",
    "evaluation_demo",
    "evaluation_smoke",
    "example_prescription_process",
    "framework_card",
    "headline_results",
    "is_structured_element",
    "table1_control_blocks",
    "transform_process",
]
