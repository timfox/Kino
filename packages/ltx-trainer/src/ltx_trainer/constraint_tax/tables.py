"""Paper table excerpts (arXiv:2605.26128)."""

from __future__ import annotations

from typing import Any


def table3_main_suite_aggregate() -> dict[str, Any]:
    """Table 3 — main GPU aggregate."""
    return {
        "prompt_json": {
            "answer_accuracy_pct": 19.7,
            "schema_validity_pct": 61.5,
            "executable_accuracy_pct": 12.0,
            "wrong_valid_schema_pct": 49.5,
            "ci_answer": [18.3, 21.1],
            "ci_valid": [59.7, 63.3],
        },
        "answer_only_schema": {
            "answer_accuracy_pct": 11.0,
            "schema_validity_pct": 100.0,
            "executable_accuracy_pct": 11.0,
            "wrong_valid_schema_pct": 88.9,
            "ci_answer": [9.9, 12.2],
            "ci_valid": [99.9, 100.0],
        },
        "delta_pts": {
            "answer_accuracy": -8.7,
            "schema_validity": 38.5,
            "executable_accuracy": -1.0,
            "wrong_valid_schema": 39.4,
        },
        "generations": 15_000,
    }


def table4_per_task_family() -> list[dict[str, Any]]:
    """Table 4 — per-family aggregates (600 gens/mode)."""
    return [
        {
            "family": "arithmetic_two_step",
            "prompt_ans": 32.2,
            "schema_ans": 5.3,
            "answer_tax_pts": 26.8,
            "wrong_valid_delta_pts": 65.0,
        },
        {
            "family": "symbolic_string",
            "prompt_ans": 0.5,
            "schema_ans": 0.0,
            "answer_tax_pts": 0.5,
            "wrong_valid_delta_pts": 21.3,
        },
        {
            "family": "object_tracking",
            "prompt_ans": 29.0,
            "schema_ans": 16.5,
            "answer_tax_pts": 12.5,
            "wrong_valid_delta_pts": 13.8,
        },
        {
            "family": "boolean_logic",
            "prompt_ans": 36.8,
            "schema_ans": 33.3,
            "answer_tax_pts": 3.5,
            "wrong_valid_delta_pts": 60.5,
        },
        {
            "family": "tool_call_argument",
            "prompt_ans": 0.0,
            "schema_ans": 0.0,
            "answer_tax_pts": 0.0,
            "wrong_valid_delta_pts": 36.7,
        },
    ]


def table5_constraint_tax_by_model() -> list[dict[str, Any]]:
    """Table 5 — tax by model / suite."""
    return [
        {
            "model": "Qwen2.5-0.5B",
            "suite": "Deterministic suite",
            "answer_tax_pts": 11.9,
            "exec_tax_pts": 6.1,
            "validity_delta_pts": 45.7,
            "wrong_valid_delta_pts": 51.8,
        },
        {
            "model": "Qwen2.5-1.5B",
            "suite": "Deterministic suite",
            "answer_tax_pts": 20.0,
            "exec_tax_pts": 12.6,
            "validity_delta_pts": 16.6,
            "wrong_valid_delta_pts": 29.2,
        },
        {
            "model": "SmolLM2-1.7B",
            "suite": "Deterministic suite",
            "answer_tax_pts": 0.0,
            "exec_tax_pts": 0.0,
            "note": "Constraint is an accuracy gain (clipped tax)",
        },
        {
            "model": "Qwen2.5-1.5B",
            "suite": "Calendar tool call",
            "exec_tax_pts": 43.5,
            "exec_tax_ci": [36.0, 51.0],
            "wrong_valid_delta_pts": 43.5,
        },
        {
            "model": "Qwen2.5-3B",
            "suite": "Boundary deterministic",
            "answer_tax_pts": 15.3,
            "answer_tax_ci": [11.2, 19.2],
            "wrong_valid_delta_pts": 31.6,
        },
    ]


def table6_calendar_analogue() -> dict[str, Any]:
    """Table 6 — calendar tool-call analogue."""
    return {
        "tool_call_prompt_json": {
            "executable_accuracy_pct": 91.5,
            "schema_validity_pct": 100.0,
            "wrong_valid_schema_pct": 8.5,
            "latency_s": 1.63,
        },
        "tool_call_delayed_packaging": {
            "executable_accuracy_pct": 91.5,
            "schema_validity_pct": 100.0,
            "note": "Deterministic re-serialize of prompt-json stage",
        },
        "tool_call_schema": {
            "executable_accuracy_pct": 48.0,
            "schema_validity_pct": 100.0,
            "wrong_valid_schema_pct": 52.0,
            "latency_s": 1.69,
        },
        "schema_minus_prompt_exec_pts": -43.5,
        "schema_minus_prompt_exec_ci": [-51.0, -36.0],
        "canonical_failure": "duration_minutes=180 for 30 min Leo meeting",
    }


def table7_calendar_failure_taxonomy() -> dict[str, int]:
    """Table 7 — primary failure counts (n=200 per mode)."""
    return {
        "tool_call_prompt_json": {
            "correct": 183,
            "wrong_duration": 8,
            "wrong_topic": 2,
            "multi_field": 2,
        },
        "tool_call_schema": {
            "correct": 96,
            "wrong_duration": 102,
            "wrong_topic": 0,
            "multi_field": 2,
        },
    }


def table8_backend_replication() -> list[dict[str, Any]]:
    return [
        {"model": "Qwen2.5-0.5B", "vllm_acc_pct": 2.7, "sglang_acc_pct": 2.7, "note": "Replicates"},
        {"model": "Qwen2.5-1.5B", "vllm_acc_pct": 11.7, "sglang_acc_pct": 11.7, "note": "Replicates"},
        {
            "model": "SmolLM2-1.7B",
            "vllm_acc_pct": 18.7,
            "sglang_acc_pct": 25.7,
            "note": "Backend-sensitive",
        },
    ]


def table9_boundary_3b() -> dict[str, Any]:
    return {
        "prompt_json": {"answer_pct": 39.0, "valid_pct": 82.6, "wrong_valid_pct": 43.6},
        "answer_only_schema": {"answer_pct": 23.7, "valid_pct": 98.9, "wrong_valid_pct": 75.2},
        "delta_answer_pts": -15.3,
        "delta_wrong_valid_pts": 31.6,
    }


def table10_expanded_interface() -> list[dict[str, Any]]:
    return [
        {"mode": "prompt_json", "answer_pct": 31.5, "valid_pct": 70.0, "exec_pct": 24.5},
        {"mode": "answer_only_schema", "answer_pct": 26.8, "valid_pct": 99.0, "exec_pct": 26.8},
        {"mode": "rationale_answer_schema", "answer_pct": 36.5, "valid_pct": 97.8, "exec_pct": 36.5},
        {"mode": "delayed_constraint", "answer_pct": 40.7, "valid_pct": 100.0, "exec_pct": 40.7},
    ]


def headline_results() -> dict[str, Any]:
    return {
        "core_claim": "Hard schemas improve validity but can reduce semantic correctness on SLMs",
        "main_suite": "100% validity vs 11% answer acc under answer_only_schema; 88.9% wrong-valid-schema",
        "calendar": "Both 100% valid; hard schema loses 43.5 pts executable accuracy",
        "pattern": "Reason free, constrain late (delayed_constraint best in expanded study)",
        "reporting": "Track schema validity, answer accuracy, executable accuracy, wrong-valid-schema separately",
    }
