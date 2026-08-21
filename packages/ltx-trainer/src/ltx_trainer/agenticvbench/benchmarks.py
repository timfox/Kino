"""AgenticVBench reference tables (paper Section 3)."""

from __future__ import annotations

from typing import Any

# Best (model, harness) per family — Figure 2 narrative (GPT-5.5 + Codex dominant).
BEST_STACK_BY_FAMILY = {
    "assembly": 0.38,
    "repair": 0.30,
    "sequencing": 0.25,  # OpenClaw wins Sequencing in paper
    "repurpose": 0.30,
}

HUMAN_REFERENCE = {
    "assembly": 0.81,
    "repair": 0.95,
    "sequencing": 0.95,
    "repurpose": 0.95,
}

# GPT-5.5 harness spread (Figure 4a) — Codex / OpenCode / OpenClaw on Assembly.
GPT55_ASSEMBLY_BY_HARNESS = {
    "codex_cli": 0.38,
    "opencode": 0.37,
    "openclaw": 0.18,
}

# Qwen3-VL Assembly: OpenCode vs OpenClaw (Figure 4b).
QWEN_ASSEMBLY_HARNESS = {
    "opencode": 0.009,
    "openclaw": 0.073,
}

FAILURE_MODES_REPURPOSE_REPAIR = {
    "repurpose": [
        {"reason": "Long-context information loss", "pct": 83},
        {"reason": "Temporal reasoning", "pct": 1},
        {"reason": "Modality misalignment", "pct": 10},
        {"reason": "Hallucinated grounding", "pct": 6},
    ],
    "repair": [
        {"reason": "Long-context information loss", "pct": 0},
        {"reason": "Temporal reasoning", "pct": 65},
        {"reason": "Modality misalignment", "pct": 24},
        {"reason": "Hallucinated grounding", "pct": 11},
    ],
}

ABLATION_LIFTS_PP = {
    "repair_oracle_localization": 13,
    "sequencing_oracle_narrative": 22,
    "assembly_strip_description": -27,
    "assembly_strip_camera_movement": -1,
    "repurpose_editor_reference_doc": 23,
}

HARNESS_PINNED = [
    {"harness": "claude_code", "package": "@anthropic-ai/claude-code", "version": "2.1.129"},
    {"harness": "codex_cli", "package": "@openai/codex", "version": "0.128.0"},
    {"harness": "gemini_cli", "package": "@google/gemini-cli", "version": "0.41.1"},
    {"harness": "opencode", "package": "opencode-ai", "version": "1.14.39"},
    {"harness": "openclaw", "package": "openclaw", "version": "2026.5.4"},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "scale": {
            "tasks": 100,
            "assembly": 18,
            "repair": 18,
            "sequencing": 28,
            "repurpose": 36,
            "experts": 20,
            "rollout_reps": 3,
            "model_harness_combos": 20,
        },
        "best_stack_by_family": BEST_STACK_BY_FAMILY,
        "human_reference": HUMAN_REFERENCE,
        "best_overall": 0.31,
        "gpt55_assembly_harness_spread": GPT55_ASSEMBLY_BY_HARNESS,
        "qwen_assembly_harness": QWEN_ASSEMBLY_HARNESS,
        "failure_modes": FAILURE_MODES_REPURPOSE_REPAIR,
        "ablation_lifts_pp": ABLATION_LIFTS_PP,
        "harness_versions": HARNESS_PINNED,
        "total_eval_cost_usd_estimate": 7477,
    }
