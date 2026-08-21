"""Evaluation harness — mini-swe-agent (fixed cross-model bash tool)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.benchmarks import HARNESS_PILOT_PASS
from ltx_trainer.deepswe.constants import HARNESS_NAME


def harness_card() -> dict[str, Any]:
    return {
        "name": HARNESS_NAME,
        "tooling": "single bash tool, shared system prompt",
        "rationale": (
            "No per-vendor apply_patch / str_replace primitives; "
            "leaderboard reflects model capability not product scaffolding."
        ),
        "upstream": "SWE-bench authors",
        "native_alternatives": ("Codex CLI", "Claude Code", "Cursor", "Gemini CLI"),
        "pilot_tasks": 10,
        "pilot_note": "mini-swe-agent matches or beats native on pilot slice",
    }


def harness_pilot_comparison() -> dict[str, Any]:
    rows = []
    for model, stats in HARNESS_PILOT_PASS.items():
        mini = stats["mini_swe_agent"]
        native = stats["native"]
        rows.append(
            {
                "model": model,
                "mini_swe_pass_pct": mini,
                "native_pass_pct": native,
                "mini_minus_native_pp": mini - native,
                "median_output_tokens_k": stats.get("median_output_tokens_k"),
            }
        )
    return {"harness": HARNESS_NAME, "pilot": rows}
