"""Task families and calendar executable checker (Appendix A)."""

from __future__ import annotations

from typing import Any


def normalize_text(x: Any) -> str:
    return str(x).strip().lower()


def calendar_exec_ok(obj: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Appendix A.3 executable checker (title not semantically scored)."""
    if obj.get("tool") != "create_calendar_event":
        return False
    args = obj.get("arguments") or {}
    return (
        normalize_text(args.get("date")) == normalize_text(expected["date"])
        and normalize_text(args.get("start_time")) == normalize_text(expected["start_time"])
        and int(args.get("duration_minutes", -1)) == int(expected["duration_minutes"])
        and normalize_text(args.get("attendee")) == normalize_text(expected["attendee"])
        and normalize_text(args.get("topic")) == normalize_text(expected["topic"])
    )


CALENDAR_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool", "arguments"],
    "properties": {
        "tool": {"const": "create_calendar_event"},
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "title",
                "date",
                "start_time",
                "duration_minutes",
                "attendee",
                "topic",
            ],
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                "start_time": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
                "duration_minutes": {"type": "integer", "minimum": 1},
                "attendee": {"type": "string"},
                "topic": {"type": "string"},
            },
        },
    },
}


TASK_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "arithmetic_two_step": "Box arithmetic with trace: initial_total, add_red, remove_blue, final_total",
    "symbolic_string": "Concatenate last letters of words with per-word trace",
    "object_tracking": "Key swaps among persons; who holds key at end",
    "boolean_logic": "AND/OR/NOT over booleans with trace ops",
    "tool_call_argument": "Canonical tool+arguments JSON for calendar scheduling (wrapper in main suite)",
}

OUTPUT_MODE_DESCRIPTIONS: dict[str, str] = {
    "freeform": "Verbose step-by-step response",
    "freeform_direct": "Answer-only final line",
    "freeform_brief_reasoning": "Short scratchpad, final answer last",
    "prompt_json": "JSON requested only by prompt",
    "final_only_regex": "Regex-constrained final answer",
    "answer_only_schema": "Minimal answer JSON schema (hard decode)",
    "rationale_answer_schema": "JSON with rationale + answer fields",
    "typed_trace_schema": "Typed JSON trace with final answer",
    "delayed_constraint": "Reason first, package answer second",
}
