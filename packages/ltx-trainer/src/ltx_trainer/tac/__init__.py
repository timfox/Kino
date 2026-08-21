"""Thinking as Compression (TaC / TaC-C) — arXiv:2605.28713."""

from ltx_trainer.tac.baselines import (
    DATASET_STATS,
    LOCOMO_4X,
    PILOT_STUDY,
    TABLE1_LLAMA_4X,
    TABLE1_LLAMA_8X,
    TABLE1_QWEN3_4X,
    TABLE1_QWEN3_8X,
    TABLE3_TRANSFER_8X,
    TABLE5_ABLATION,
    THINKER_SCALE_EM,
    TRACE_EXAMPLES,
)
from ltx_trainer.tac.config import TaCConfig
from ltx_trainer.tac.metrics import exact_match, f1_score, utility_reward
from ltx_trainer.tac.mock import evaluation_smoke
from ltx_trainer.tac.pipeline import (
    TaCExample,
    benchmarks_bundle,
    demo_film_example,
    evaluation_demo,
    framework_card,
    knowledge_card,
    tac_c_select_trace,
    tac_vanilla,
)
from ltx_trainer.tac.prompts import answerer_prompt, thinker_prompt, wrap_thinking
from ltx_trainer.tac.rewards import (
    RewardBreakdown,
    budget_reward,
    detect_hack,
    extract_thinking,
    format_reward,
    grpo_group_advantages,
    total_reward,
)

__all__ = [
    "DATASET_STATS",
    "LOCOMO_4X",
    "PILOT_STUDY",
    "RewardBreakdown",
    "TABLE1_LLAMA_4X",
    "TABLE1_LLAMA_8X",
    "TABLE1_QWEN3_4X",
    "TABLE1_QWEN3_8X",
    "TABLE3_TRANSFER_8X",
    "TABLE5_ABLATION",
    "THINKER_SCALE_EM",
    "TRACE_EXAMPLES",
    "TaCConfig",
    "TaCExample",
    "answerer_prompt",
    "benchmarks_bundle",
    "budget_reward",
    "demo_film_example",
    "detect_hack",
    "evaluation_demo",
    "evaluation_smoke",
    "exact_match",
    "extract_thinking",
    "f1_score",
    "format_reward",
    "framework_card",
    "grpo_group_advantages",
    "knowledge_card",
    "tac_c_select_trace",
    "tac_vanilla",
    "thinker_prompt",
    "total_reward",
    "utility_reward",
    "wrap_thinking",
]
