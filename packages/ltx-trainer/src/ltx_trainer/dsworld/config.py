"""DSWorld — data science world model for autonomous agents (arXiv:2607.15901)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15901"
PAPER_TITLE = "DSWorld: A Data Science World Model for Efficient Autonomous Agents"
PAPER_SYSTEM = "DSWorld"
PAPER_AUTHORS = "Zherui Yang, Fan Liu, Hao Liu (HKUST Guangzhou)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_GITHUB = "https://anonymous.4open.science/r/DSWorld"
BENCHMARK = (
    "Predict-before-Execute + 540 synthetic transition tasks + MLE-Bench Lite; "
    "DSWorld-8K training trajectories"
)

COMPONENTS = (
    "state_constructor",
    "cost_aware_router",
    "compiler",
    "llm_simulator",
    "reflective_world_model_optimization",
)

ROUTE_MODES = ("execute", "simulate")


@dataclass
class DSWorldConfig:
    """Runtime knobs for the CPU stub."""

    compiler_timeout_s: float = 30.0
    simulate_cost_threshold: float = 5.0  # estimated minutes → simulate if above
    n_reflect_rollouts: int = 8
    sft_epochs: int = 5
    rl_steps: int = 200
    backbone: str = "Qwen3-8B"
