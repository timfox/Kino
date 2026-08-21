"""Local run plan for DeepSWE (Harbor / mini-swe-agent upstream)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.config import DeepSWEConfig
from ltx_trainer.deepswe.constants import GITHUB_REPO, HARNESS_NAME


def run_plan(cfg: DeepSWEConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeepSWEConfig()
    return {
        "benchmark": cfg.name,
        "github": GITHUB_REPO,
        "website": cfg.website,
        "harness": HARNESS_NAME,
        "steps": [
            "git clone https://github.com/datacurve-ai/deep-swe",
            "Follow upstream README for Harbor task format + container images",
            f"Run agent via {HARNESS_NAME} with pinned model endpoint",
            "Aggregate pass@1 with behavioral verifiers per task",
            "Compare to publication leaderboard in ltx_trainer.deepswe.benchmarks",
        ],
        "gopex_hook": cfg.ltx_hook,
        "env_hints": {
            "GOPEX_DEEPSWE_ROOT": "optional checkout path",
            "mini_swe_agent": "pip install per upstream deep-swe repo",
        },
        "notes": [
            "Tasks use shallow clone — no gold commit in workspace.",
            "Verifiers are behavioral; do not assume PR-shaped patches.",
            "For GOPEX stack QA, use same harness on internal repo tasks.",
        ],
    }
