"""Paper anchors and table stubs for DSWorld (arXiv:2607.15901)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "DSWorld",
    "n_train_trajectories": 8000,
    "rl_speedup_x": 14.0,
    "inference_speedup_lo": 3.0,
    "inference_speedup_hi": 6.0,
    "transition_gain_vs_o4mini": 0.356,
    "esp": 0.950,
    "etp": 0.922,
    "ers": 0.871,
    "ekm": 0.575,
    "pp": 0.856,
    "pr": 0.518,
    "avg": 0.781,
    "backbone": "Qwen3-8B",
    "github": "https://anonymous.4open.science/r/DSWorld",
}

# Table 1 average scores
TABLE_1_AVG: dict[str, float] = {
    "Llama-3.1-8B": 0.379,
    "Qwen3-8B": 0.555,
    "DeepSeek-3.2": 0.516,
    "GPT-4o": 0.518,
    "o4-mini": 0.576,
    "Llama-3.1-8B-sft": 0.726,
    "Llama-3.1-8B-grpo": 0.747,
    "Qwen3-8B-sft": 0.763,
    "Qwen3-8B-grpo": 0.771,
    "DSWorld": 0.781,
}

TABLE_1_DSWORLD_ROW: dict[str, float] = {
    "ESP": 0.950,
    "ETP": 0.922,
    "ERS": 0.871,
    "EKM": 0.575,
    "PP": 0.856,
    "PR": 0.518,
    "AVG": 0.781,
}

# Table 2: RL training with simulators (Qwen3-8B)
TABLE_2_TRAINING: list[dict[str, Any]] = [
    {"simulator": "-", "score": 13.80, "time_min": None},
    {"simulator": "DeepSeek 3.2", "score": 10.86, "time_min": 3854},
    {"simulator": "Compiler", "score": 18.11, "time_min": None},  # wall-clock dominated; fig shows slow
    {"simulator": "DSWorld", "score": 17.67, "time_min": 277},
]

# Approximate inference times from Table 3 (AIDE / Qwen3-8B)
TABLE_3_AIDE_QWEN: list[dict[str, Any]] = [
    {"executer": "Compiler", "score": 10.7, "time_s": 4102},
    {"executer": "DeepSeek 3.2", "score": 7.21, "time_s": 806},
    {"executer": "DSWorld", "score": 10.58, "time_s": 676},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_avg": dict(TABLE_1_AVG),
        "table_1_dsworld": dict(TABLE_1_DSWORLD_ROW),
        "table_2_training": list(TABLE_2_TRAINING),
        "table_3_aide_qwen": list(TABLE_3_AIDE_QWEN),
    }
