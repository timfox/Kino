"""Framework card and benchmark bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.constraint_tax.config import ConstraintTaxConfig
from ltx_trainer.constraint_tax.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.constraint_tax.mock import evaluation_smoke
from ltx_trainer.constraint_tax.tables import (
    headline_results,
    table10_expanded_interface,
    table3_main_suite_aggregate,
    table4_per_task_family,
    table5_constraint_tax_by_model,
    table6_calendar_analogue,
    table7_calendar_failure_taxonomy,
    table8_backend_replication,
    table9_boundary_3b,
)
from ltx_trainer.constraint_tax.tasks import (
    CALENDAR_TOOL_SCHEMA,
    OUTPUT_MODE_DESCRIPTIONS,
    TASK_FAMILY_DESCRIPTIONS,
)


def framework_card(cfg: ConstraintTaxConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ConstraintTaxConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "author": cfg.author,
        "problem": (
            "Sub-3B SLMs under hard structured decoding: schema validity can rise while "
            "answer/executable accuracy falls — wrong answer, valid schema."
        ),
        "constraint_tax": {
            "absolute": "Tax = max(0, Acc(baseline) − Acc(constrained))",
            "normalized": "Tax_norm = Tax / max(ε, Acc(baseline))",
            "baseline_modes": ["prompt_json", "freeform", "mode-specific"],
        },
        "metrics": [
            "schema_validity",
            "answer_accuracy",
            "executable_accuracy",
            "trace_correctness",
            "wrong_valid_schema_rate",
            "latency",
            "output_tokens",
            "structural_overhead",
        ],
        "harness": {
            "main_generations": cfg.main_generations,
            "task_families": list(cfg.task_families),
            "backends": list(cfg.backends),
            "checkpoints": list(cfg.main_checkpoints),
        },
        "task_families": TASK_FAMILY_DESCRIPTIONS,
        "output_modes": OUTPUT_MODE_DESCRIPTIONS,
        "design_pattern": "Reason free, constrain late (delayed_constraint, rationale schemas)",
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table3_main_aggregate": table3_main_suite_aggregate(),
        "table4_per_task": table4_per_task_family(),
        "table5_tax_by_model": table5_constraint_tax_by_model(),
        "table6_calendar": table6_calendar_analogue(),
        "table7_calendar_failures": table7_calendar_failure_taxonomy(),
        "table8_backends": table8_backend_replication(),
        "table9_boundary_3b": table9_boundary_3b(),
        "table10_expanded_interface": table10_expanded_interface(),
        "calendar_schema_excerpt": CALENDAR_TOOL_SCHEMA,
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headlines": headline_results()}
