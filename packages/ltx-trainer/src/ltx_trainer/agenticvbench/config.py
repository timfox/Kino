"""AgenticVBench configuration (arXiv:2605.27705)."""

from __future__ import annotations

from dataclasses import dataclass, field


TASK_FAMILIES = ("assembly", "repair", "sequencing", "repurpose")
HARNESS_IDS = ("claude_code", "codex_cli", "gemini_cli", "opencode", "openclaw")
REPURPOSE_PILLARS = ("format", "visual", "narrative", "sound", "penalty")


@dataclass
class AgenticVBenchConfig:
    arxiv: str = "2605.27705"
    website: str = "https://agenticvbench.com"
    num_tasks: int = 100
    tasks_per_family: dict[str, int] = field(
        default_factory=lambda: {
            "repurpose": 36,
            "sequencing": 28,
            "repair": 18,
            "assembly": 18,
        }
    )
    rollout_reps: int = 3
    max_agent_iterations: int = 200
    per_task_timeout_s: int = 1800
    expert_pool_size: int = 20
    human_reference: dict[str, float] = field(
        default_factory=lambda: {
            "assembly": 0.81,
            "repair": 0.95,
            "sequencing": 0.95,
            "repurpose": 0.95,
        }
    )
    best_stack_overall: float = 0.31
    models_evaluated: tuple[str, ...] = (
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "gpt-5.5",
        "gpt-5.4-mini",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "qwen3-vl-235b-a22b-instruct",
    )
