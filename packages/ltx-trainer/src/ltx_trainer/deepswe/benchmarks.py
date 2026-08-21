"""Paper anchors and leaderboard tables — DeepSWE (Datacurve 2026)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.constants import (
    GITHUB_REPO,
    LANGUAGE_COUNTS,
    MEAN_FILES_EDITED,
    MEAN_PROMPT_CHARS,
    MEAN_REF_LINES_ADDED,
    N_REPOS,
    N_TASKS,
    VERIFIER_AUDIT,
    WEBSITE,
)

# Leaderboard pass rates (%, publication snapshot May 26 2026; mini-swe-agent harness)
LEADERBOARD_PASS_RATE: dict[str, dict[str, float | str]] = {
    "gpt-5.5[xhigh]": {"pass_pct": 70.0, "margin_pp": 4.0},
    "gpt-5.4[xhigh]": {"pass_pct": 56.0, "margin_pp": 5.0},
    "claude-opus-4.7[max]": {"pass_pct": 54.0, "margin_pp": 5.0},
    "claude-sonnet-4.6[high]": {"pass_pct": 32.0, "margin_pp": 4.0},
    "gemini-3.5-flash[medium]": {"pass_pct": 28.0, "margin_pp": 4.0},
    "gpt-5.4-mini[xhigh]": {"pass_pct": 24.0, "margin_pp": 4.0},
    "kimi-k2.6": {"pass_pct": 24.0, "margin_pp": 4.0},
    "mimo-v2.5-pro": {"pass_pct": 19.0, "margin_pp": 4.0},
    "glm-5.1": {"pass_pct": 18.0, "margin_pp": 4.0},
    "gemini-3.1-pro": {"pass_pct": 10.0, "margin_pp": 3.0},
    "deepseek-v4-pro": {"pass_pct": 8.0, "margin_pp": 2.0},
    "gemini-3-flash": {"pass_pct": 5.0, "margin_pp": 2.0},
}

# Same models on publicly reported SWE-Bench Pro (blog comparison chart)
SWE_BENCH_PRO_PASS: dict[str, float] = {
    "claude-opus-4.7": 64.0,
    "gpt-5.5": 59.0,
    "gpt-5.4": 58.0,
    "claude-sonnet-4.6": 54.0,
    "gpt-5.4-mini": 54.0,
    "gemini-3.1-pro": 46.0,
    "claude-haiku-4.5": 39.0,
    "gemini-3-flash": 35.0,
}

DEEPSWE_PASS_NORM: dict[str, float] = {
    "claude-opus-4.7": 54.0,
    "gpt-5.5": 70.0,
    "gpt-5.4": 56.0,
    "claude-sonnet-4.6": 32.0,
    "gpt-5.4-mini": 24.0,
    "gemini-3.1-pro": 10.0,
    "gemini-3-flash": 5.0,
    "claude-haiku-4.5": 0.0,  # not on DeepSWE chart; 0 placeholder
}

# mini-swe-agent vs native harness pilot (10 SWE-Bench Pro tasks, blog)
HARNESS_PILOT_PASS: dict[str, dict[str, float]] = {
    "claude-opus-4.7": {"mini_swe_agent": 50.0, "native": 40.0, "median_output_tokens_k": 59.0},
    "gpt-5.5": {"mini_swe_agent": 40.0, "native": 40.0, "median_output_tokens_k": 16.0},
    "gemini-3.1-pro": {"mini_swe_agent": 40.0, "native": 20.0, "median_output_tokens_k": 18.0},
}

# Cost-shaped medians (blog; subset of models)
MEDIAN_OUTPUT_TOKENS: dict[str, int] = {
    "gpt-5.5": 47_000,
    "gpt-5.4": 90_000,
    "claude-opus-4.7": 149_000,
}
MEDIAN_WALL_CLOCK_MIN: dict[str, int] = {
    "gpt-5.5": 20,
    "gemini-3.5-flash": 15,
}
MEDIAN_COST_USD: dict[str, float] = {
    "gpt-5.4": 3.3,
    "gpt-5.5": 5.8,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "website": WEBSITE,
        "github": GITHUB_REPO,
        "n_tasks": N_TASKS,
        "n_repos": N_REPOS,
        "language_counts": LANGUAGE_COUNTS,
        "mean_prompt_chars": MEAN_PROMPT_CHARS,
        "mean_ref_lines_added": MEAN_REF_LINES_ADDED,
        "mean_files_edited": MEAN_FILES_EDITED,
        "leaderboard_pass_rate": LEADERBOARD_PASS_RATE,
        "swe_bench_pro_pass": SWE_BENCH_PRO_PASS,
        "deepswe_pass_comparison": DEEPSWE_PASS_NORM,
        "verifier_audit": VERIFIER_AUDIT,
        "harness_pilot": HARNESS_PILOT_PASS,
        "median_output_tokens": MEDIAN_OUTPUT_TOKENS,
        "median_wall_clock_min": MEDIAN_WALL_CLOCK_MIN,
        "median_cost_usd": MEDIAN_COST_USD,
        "separation_span_pp": {
            "deepswe_leaderboard": 70.0,
            "swe_bench_pro_reported": 30.0,
        },
    }
