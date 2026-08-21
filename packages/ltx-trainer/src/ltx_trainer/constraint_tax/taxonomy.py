"""Error taxonomy (Sec. 4.4)."""

from __future__ import annotations

from typing import Literal

ErrorClass = Literal[
    "correct_valid",
    "invalid_json",
    "parse_failure_freeform",
    "schema_validation_error",
    "trace_answer_contradiction",
    "wrong_answer_valid_schema",
]


def classify_record(
    *,
    schema_valid: bool,
    answer_correct: bool,
    executable_ok: bool,
    parse_ok: bool,
    trace_ok: bool | None = None,
    freeform: bool = False,
) -> ErrorClass:
    if schema_valid and answer_correct and executable_ok:
        return "correct_valid"
    if not parse_ok and freeform:
        return "parse_failure_freeform"
    if not parse_ok:
        return "invalid_json"
    if schema_valid and not answer_correct:
        if trace_ok is False:
            return "trace_answer_contradiction"
        return "wrong_answer_valid_schema"
    if not schema_valid:
        return "schema_validation_error"
    return "wrong_answer_valid_schema"
