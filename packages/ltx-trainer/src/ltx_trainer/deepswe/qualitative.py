"""Qualitative failure-mode taxonomy (blog trajectory analysis)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.constants import FAILURE_TAGS


def failure_mode_taxonomy() -> dict[str, str]:
    return {
        "true_positive": "Verifier pass; patch implements requested behavior.",
        "true_negative": "Verifier fail; patch does not satisfy task.",
        "false_positive": "Verifier pass; analyzer judges patch incorrect.",
        "false_negative": "Verifier fail; analyzer judges patch correct.",
        "MISSED_REQUIREMENT": (
            "Near-correct solution missing a crucial detail (common on multi-branch prompts)."
        ),
        "CHEATED": "Recovered gold solution from .git history (SWE-Bench Pro containers).",
        "TEST_MISMATCH": "Verifier failed but implementation reasonable (often Pro inherited tests).",
    }


def model_family_notes() -> dict[str, Any]:
    return {
        "claude": {
            "missed_requirement": "Often implements one branch of parallel requirements (sync not async, etc.).",
            "cheated_swe_bench_pro": ">12% CHEATED on reviewed Pro rollouts via git log/show gold commit.",
            "environment_attentive": "Uses git log to reconcile prompt/repo mismatch.",
        },
        "gpt": {
            "missed_requirement": "Lowest MISSED_REQUIREMENT rate on DeepSWE in blog analysis.",
            "instruction_literal": "Reads prompt and repo contract literally; stable across trials.",
            "self_test": "Writes new tests in project framework >80% of DeepSWE runs unprompted.",
        },
        "gemini": {
            "verification": "Weaker configurations skip tests more often.",
        },
        "swe_bench_pro_prompt": (
            "Template tells agent not to modify tests → test-writing drops to 3–28% on Pro vs much higher on DeepSWE."
        ),
    }


def swe_bench_pro_failure_patterns() -> list[dict[str, str]]:
    return [
        {
            "pattern": "git_history_gold_leak",
            "pro": "Full .git in container; agents git show gold hash",
            "deepswe": "Shallow clone; no gold commit in workspace",
        },
        {
            "pattern": "weak_gold_tests",
            "pro": "Stubs pass paths gold PR exercised only",
            "deepswe": "End-to-end behavioral assertions",
        },
        {
            "pattern": "private_helper_import",
            "pro": "Tests import maintainer-only symbols",
            "deepswe": "Public observable outputs only",
        },
        {
            "pattern": "fixture_not_restored",
            "pro": "git checkout test file without fixture data",
            "deepswe": "Author-owned verifier bundles fixtures",
        },
        {
            "pattern": "unrelated_gold_tests",
            "pro": "Extra tests fail correct but broader patches",
            "deepswe": "Scoped behavioral + regression checks",
        },
    ]


def qualitative_bundle() -> dict[str, Any]:
    return {
        "tags": list(FAILURE_TAGS),
        "taxonomy": failure_mode_taxonomy(),
        "model_families": model_family_notes(),
        "swe_bench_pro_patterns": swe_bench_pro_failure_patterns(),
        "sampled_tasks_per_benchmark": 30,
        "trials_per_task": 3,
        "agent_configurations": 9,
    }
