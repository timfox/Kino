"""Behavioral verifier model for DeepSWE-style task grading (CPU stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BehavioralTest:
    """Single observable-behavior check (public API / output only)."""

    name: str
    check: Callable[[dict[str, Any]], bool]
    description: str = ""


@dataclass
class DeepSWETaskSpec:
    """Task artifacts: prompt, verifier tests, reference solution (review only)."""

    task_id: str
    repo: str
    language: str
    prompt: str
    tests: list[BehavioralTest] = field(default_factory=list)
    regression_test_ids: tuple[str, ...] = ()

    def grade(self, observable_state: dict[str, Any]) -> dict[str, Any]:
        """Run behavioral tests + optional regression flag."""
        results = []
        for t in self.tests:
            try:
                ok = bool(t.check(observable_state))
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = str(exc)
            else:
                err = None
            results.append({"name": t.name, "ok": ok, "error": err})
        passed = all(r["ok"] for r in results) and observable_state.get("regression_ok", True)
        return {
            "task_id": self.task_id,
            "passed": passed,
            "tests": results,
            "n_tests": len(results),
            "verifier_kind": "behavioral_public_api",
        }


def example_task_boa_cancellation() -> DeepSWETaskSpec:
    """Blog example: Rust Boa eval cancellation — Promise.then must respect cancel."""

    def no_orphan_then_callbacks(state: dict[str, Any]) -> bool:
        return state.get("then_callbacks_without_handle", 0) == 0

    def unrelated_jobs_run(state: dict[str, Any]) -> bool:
        return state.get("unrelated_jobs_completed", 0) >= 1

    return DeepSWETaskSpec(
        task_id="boa-evaluation-cancellation",
        repo="boa-dev/boa",
        language="rust",
        prompt="Add cancellation support to Boa evaluations; cancelled work must not run.",
        tests=[
            BehavioralTest(
                "cancel_stops_queued_js_work",
                lambda s: s.get("cancelled_eval_pending_jobs", 1) == 0,
                "No not-yet-started jobs after cancel",
            ),
            BehavioralTest(
                "promise_then_respects_cancel",
                no_orphan_then_callbacks,
                "Promise.then callbacks enqueued with cancellation handle",
            ),
            BehavioralTest(
                "unrelated_jobs_ok",
                unrelated_jobs_run,
                "Unrelated jobs still complete",
            ),
        ],
        regression_test_ids=("boa_smoke", "promise_basic"),
    )


def verifier_audit_summary() -> dict[str, Any]:
    """Published LLM-judge vs verifier disagreement rates."""
    from ltx_trainer.deepswe.constants import VERIFIER_AUDIT

    return {
        "reviewed_rollouts": {"deepswe": 735, "swe_bench_pro": 789},
        "rates": VERIFIER_AUDIT,
        "design": (
            "DeepSWE verifiers are purpose-written from the task description; "
            "SWE-Bench Pro often inherits merged-PR test suites."
        ),
    }
