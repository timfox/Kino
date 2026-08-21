"""MCBench multicontext omni safety — Luong et al., arXiv:2606.05177."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McbenchConfig:
    paper_arxiv: str = "arXiv:2606.05177"
    title: str = "MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models"
    framework: str = "MCBench"
    venue: str = "arXiv"

    total_scenarios: int = 1196
    modalities: tuple[str, ...] = ("vision", "audio", "speech")
    judge_model: str = "GPT-4o"
    eval_runs: int = 5

    # Table 2 category counts
    physical_harm: int = 300
    social_harm: int = 300
    illegal_harm: int = 296
    property_damage: int = 300

    # Table 3 average accuracy (CoT, LLM-as-judge)
    qwen_omni_3b_avg: float = 64.5
    qwen_omni_7b_avg: float = 55.2
    gemini_flash_avg: float = 64.4
    random_baseline: float = 50.0

    # Category-specific (Gemini unsafe subset highlights)
    gemini_social_unsafe_acc: float = 44.0

    # Fig 7 perception alignment (illegal harm, Gemini)
    gemini_perception_illegal: float = 0.698

    # Table 4 setting 2 deltas (Gemini)
    gemini_safe_drop_setting2: float = 16.83
    gemini_unsafe_gain_setting2: float = 29.17
    qwen3b_safe_drop_setting2: float = 46.0
