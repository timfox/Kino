"""Constraint Tax SLM structured-output benchmark stub (arXiv:2605.26128)."""

from ltx_trainer.constraint_tax.config import ConstraintTaxConfig
from ltx_trainer.constraint_tax.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.constraint_tax.metrics import (
    aggregate_from_records,
    constraint_tax,
    constraint_tax_normalized,
)
from ltx_trainer.constraint_tax.mock import evaluation_smoke
from ltx_trainer.constraint_tax.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.constraint_tax.tables import headline_results
from ltx_trainer.constraint_tax.tasks import calendar_exec_ok
from ltx_trainer.constraint_tax.taxonomy import classify_record

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "ConstraintTaxConfig",
    "aggregate_from_records",
    "benchmarks_bundle",
    "calendar_exec_ok",
    "classify_record",
    "constraint_tax",
    "constraint_tax_normalized",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
]
