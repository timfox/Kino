"""Repository selection, task construction, and QA methodology."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.constants import (
    LANGUAGES,
    MIN_GITHUB_STARS,
    N_REPOS,
    N_TASKS,
    QA_DIMENSIONS,
)


def repository_selection_criteria() -> dict[str, Any]:
    return {
        "public": True,
        "actively_maintained": True,
        "min_github_stars": MIN_GITHUB_STARS,
        "permissive_license": True,
        "pinned_commit": True,
        "languages": list(LANGUAGES),
        "median_tasks_per_repo": 1,
        "n_repos": N_REPOS,
    }


def task_construction_pipeline() -> list[dict[str, str]]:
    return [
        {"step": "prompt", "artifact": "behavior-focused short developer message"},
        {"step": "verifier", "artifact": "new behavioral tests on public APIs"},
        {"step": "reference_solution", "artifact": "authored from scratch; not used at grade time"},
        {"step": "verifier_stability", "artifact": "3× run during authoring; flaky verifiers revised"},
        {"step": "regression", "artifact": "existing repo tests + author regression tests"},
    ]


def quality_assurance_rubric() -> dict[str, str]:
    return {
        "prompt_verifier_bijection": (
            "Verifier tests exactly the prompt behavior — no extra requirements (FN) "
            "and no missing checks (FP)."
        ),
        "acceptance_breadth": (
            "Any reasonable implementation passing observable behavior is accepted, "
            "not only the reference solution shape."
        ),
        "realism": (
            "Prompt reads like a real agent message; task is a plausible maintainer contribution."
        ),
        "environment_cleanliness": (
            "Failures reflect model capability, not flaky infra or dependency breakage."
        ),
    }


def methodology_card() -> dict[str, Any]:
    return {
        "n_tasks": N_TASKS,
        "novel_tasks": True,
        "not_merged_upstream": True,
        "shallow_clone_no_gold_hash": True,
        "contamination_mitigation": (
            "Tasks written from scratch; not adapted from public PRs; never merged upstream."
        ),
        "repository_selection": repository_selection_criteria(),
        "task_pipeline": task_construction_pipeline(),
        "qa_dimensions": list(QA_DIMENSIONS),
        "qa_rubric": quality_assurance_rubric(),
        "review": "LLM-assisted analysis + independent human review before acceptance",
    }
