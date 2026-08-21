"""Paper table anchors for Role-Agent (arXiv:2606.10917)."""

from __future__ import annotations

from typing import Any


def table1_alfworld_webshop_qwen15b() -> dict[str, Any]:
    """Table 1 — Qwen2.5-1.5B-Instruct RL methods."""
    return {
        "GiGPO": {"alfworld_all": 86.7, "webshop_score": 83.1, "webshop_succ": 65.0},
        "Role-Agent": {"alfworld_all": 90.9, "webshop_score": 87.7, "webshop_succ": 71.9},
        "GRPO": {"alfworld_all": 72.8, "webshop_score": 75.8, "webshop_succ": 56.8},
    }


def table2_search_qa_qwen3b() -> dict[str, Any]:
    """Table 2 — search-augmented QA (Qwen2.5-3B-Instruct)."""
    return {
        "GiGPO": {"avg": 42.1, "HotpotQA": 36.9, "2Wiki": 37.0, "MuSiQue": 12.6, "Bamboogle": 64.1},
        "Role-Agent": {"avg": 45.8, "HotpotQA": 38.8, "2Wiki": 45.2, "MuSiQue": 17.8, "Bamboogle": 68.4},
    }


def table3_ablation_qwen15b() -> dict[str, Any]:
    """Table 3 — component ablation."""
    return {
        "Role-Agent": {"alfworld": 90.9, "webshop": 71.9, "avg": 81.4},
        "w/o AIW": {"alfworld": 87.5, "webshop": 66.9, "avg": 77.2},
        "w/o predictive reward": {"alfworld": 88.0, "webshop": 68.3, "avg": 78.2},
        "GiGPO": {"alfworld": 86.7, "webshop": 65.0, "avg": 75.9},
    }


def table4_hyperparam_sensitivity() -> dict[str, Any]:
    return {
        "alpha_1.0": {"alfworld": 90.9, "webshop": 71.9, "avg": 81.4},
        "H_5pct_tmax": {"alfworld": 90.9, "webshop": 71.9, "avg": 81.4},
    }


def table7_stability() -> dict[str, Any]:
    return {
        "Qwen2.5-1.5B": {
            "GiGPO": {"alfworld": "86.7±0.6", "webshop": "65.0±1.1"},
            "Role-Agent": {"alfworld": "90.9±0.8", "webshop": "71.9±0.9"},
        },
        "Qwen2.5-7B": {
            "GiGPO": {"alfworld": "90.8±0.5", "webshop": "72.8±1.8"},
            "Role-Agent": {"alfworld": "93.8±0.8", "webshop": "77.1±0.6"},
        },
    }


def table5_hyperparameters() -> dict[str, Any]:
    return {
        "learning_rate": 1e-6,
        "group_size": 8,
        "state_similarity_threshold": 0.9,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "kl_coefficient": 1e-3,
        "max_interaction_step": "50/15/4",
        "total_epochs": 150,
    }


def benchmarks_bundle() -> dict[str, Any]:
    t1 = table1_alfworld_webshop_qwen15b()
    t3 = table3_ablation_qwen15b()
    return {
        "table1_qwen15b": t1,
        "table2_search_qa_qwen3b": table2_search_qa_qwen3b(),
        "table3_ablation": t3,
        "table4_sensitivity": table4_hyperparam_sensitivity(),
        "table5_hyperparameters": table5_hyperparameters(),
        "table7_stability": table7_stability(),
        "role_agent_vs_gigpo_alfworld_pp": t1["Role-Agent"]["alfworld_all"] - t1["GiGPO"]["alfworld_all"],
        "role_agent_vs_gigpo_webshop_pp": t1["Role-Agent"]["webshop_succ"] - t1["GiGPO"]["webshop_succ"],
        "ablation_aiw_drop_webshop_pp": t3["Role-Agent"]["webshop"] - t3["w/o AIW"]["webshop"],
        "ablation_both_beat_gigpo": (
            t3["w/o AIW"]["avg"] > t3["GiGPO"]["avg"] and t3["w/o predictive reward"]["avg"] > t3["GiGPO"]["avg"]
        ),
    }
