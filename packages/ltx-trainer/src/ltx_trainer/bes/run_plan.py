"""Operational run plans for reproducing paper experiments."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bes.constants import GITHUB_REPO
from ltx_trainer.bes.experiments import experiments_bundle


def replication_checklist() -> list[str]:
    return [
        f"Clone upstream: {GITHUB_REPO}",
        "Logical reasoning: Gemma-3-1B-it + K&K (2× H200, vLLM decomposer)",
        "MuSiQue: Llama-3.2-3B / 8B + E5 retriever + backward decomposer",
        "Open problems: ShinkaEvolve + GPT-5 + BES LLM evolution prompts",
        "Compare Tables 1–2 against SkyDiscover baselines",
    ]


def run_plan_bundle() -> dict[str, Any]:
    return {
        "checklist": replication_checklist(),
        "experiments": experiments_bundle(),
        "hardware_notes": {
            "post_train": "2× NVIDIA H200 (trainer + auxiliary vLLM)",
            "open_problems": "CPU node + OpenAI API ($50 cap/run)",
        },
    }
