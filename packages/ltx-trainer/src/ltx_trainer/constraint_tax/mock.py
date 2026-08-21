"""Toy paired comparison illustrating constraint tax."""

from __future__ import annotations

from typing import Any

from ltx_trainer.constraint_tax.config import ConstraintTaxConfig
from ltx_trainer.constraint_tax.metrics import (
    aggregate_from_records,
    constraint_tax,
    constraint_tax_normalized,
)
from ltx_trainer.constraint_tax.tables import table3_main_suite_aggregate, table6_calendar_analogue
from ltx_trainer.constraint_tax.taxonomy import classify_record
from ltx_trainer.constraint_tax.tasks import calendar_exec_ok


def _toy_main_suite_records() -> tuple[list[dict], list[dict]]:
    """Synthetic 20-row paired sample mirroring Table 3 qualitative split."""
    prompt_rows: list[dict] = []
    schema_rows: list[dict] = []
    # 4 correct-valid, 8 wrong-valid under schema, 8 invalid under prompt
    for i in range(20):
        prompt_valid = i % 5 != 0  # ~80% invalid -> ~20% valid band scaled
        prompt_correct = i < 4
        prompt_rows.append(
            {
                "schema_valid": prompt_valid,
                "answer_correct": prompt_correct,
                "executable_ok": prompt_correct and prompt_valid,
                "parse_ok": prompt_valid or i % 3 != 0,
                "wrong_valid_schema": prompt_valid and not prompt_correct,
            }
        )
        schema_correct = i < 2
        schema_rows.append(
            {
                "schema_valid": True,
                "answer_correct": schema_correct,
                "executable_ok": schema_correct,
                "parse_ok": True,
                "wrong_valid_schema": not schema_correct,
            }
        )
    return prompt_rows, schema_rows


def _toy_calendar_records() -> tuple[list[dict], list[dict]]:
    expected = {
        "date": "2026-05-20",
        "start_time": "14:00",
        "duration_minutes": 30,
        "attendee": "leo",
        "topic": "sync review",
    }
    prompt_rows: list[dict] = []
    schema_rows: list[dict] = []
    for i in range(20):
        prompt_obj = {
            "tool": "create_calendar_event",
            "arguments": {
                "title": "Meeting",
                "date": "2026-05-20",
                "start_time": "14:00",
                "duration_minutes": 30,
                "attendee": "Leo",
                "topic": "sync review",
            },
        }
        ok = calendar_exec_ok(prompt_obj, expected)
        if i >= 18:
            ok = False
            prompt_obj["arguments"]["duration_minutes"] = 60
        prompt_rows.append(
            {
                "schema_valid": True,
                "answer_correct": ok,
                "executable_ok": ok,
                "parse_ok": True,
                "wrong_valid_schema": not ok,
            }
        )
        schema_obj = dict(prompt_obj)
        if i < 10:
            schema_obj["arguments"] = dict(schema_obj["arguments"])
            schema_obj["arguments"]["duration_minutes"] = 180  # paper failure mode
            ok_s = False
        else:
            ok_s = calendar_exec_ok(schema_obj, expected)
        schema_rows.append(
            {
                "schema_valid": True,
                "answer_correct": ok_s,
                "executable_ok": ok_s,
                "parse_ok": True,
                "wrong_valid_schema": not ok_s,
            }
        )
    return prompt_rows, schema_rows


def evaluation_smoke(cfg: ConstraintTaxConfig | None = None) -> dict[str, Any]:
    c = cfg or ConstraintTaxConfig()
    p_rows, s_rows = _toy_main_suite_records()
    agg_p = aggregate_from_records(p_rows)
    agg_s = aggregate_from_records(s_rows)
    tax_ans = constraint_tax(agg_p.answer_accuracy, agg_s.answer_accuracy)
    tax_norm = constraint_tax_normalized(agg_p.answer_accuracy, agg_s.answer_accuracy)

    cal_p, cal_s = _toy_calendar_records()
    cal_agg_p = aggregate_from_records(cal_p)
    cal_agg_s = aggregate_from_records(cal_s)
    exec_tax = constraint_tax(cal_agg_p.executable_accuracy, cal_agg_s.executable_accuracy)

    sample_err = classify_record(
        schema_valid=True,
        answer_correct=False,
        executable_ok=False,
        parse_ok=True,
    )

    paper_t3 = table3_main_suite_aggregate()
    paper_cal = table6_calendar_analogue()

    return {
        "paper": c.paper_arxiv,
        "toy_main_suite": {
            "prompt_json": agg_p.to_dict(),
            "answer_only_schema": agg_s.to_dict(),
            "answer_tax_pts": round(tax_ans, 1),
            "answer_tax_normalized": round(tax_norm, 3),
        },
        "toy_calendar": {
            "prompt_json_exec_pct": cal_agg_p.executable_accuracy,
            "schema_exec_pct": cal_agg_s.executable_accuracy,
            "executable_tax_pts": round(exec_tax, 1),
        },
        "paper_table3_answer_tax_pts": paper_t3["delta_pts"]["answer_accuracy"],
        "paper_calendar_exec_delta_pts": paper_cal["schema_minus_prompt_exec_pts"],
        "sample_error_class": sample_err,
        "task_families": list(c.task_families),
        "output_modes_count": len(c.output_modes_full),
    }
