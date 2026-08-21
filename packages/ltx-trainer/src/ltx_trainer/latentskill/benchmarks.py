"""Paper table anchors (arXiv:2606.06087)."""
from __future__ import annotations

from typing import Any


def table1_alfworld() -> dict[str, Any]:
    return {
        "LatentSkill": {
            "seen_avg": 74.3,
            "unseen_avg": 69.4,
            "seen_step": 28.4,
            "seen_prefill_k": 0.44,
            "unseen_prefill_k": 0.44,
        },
        "In-Context Skill": {
            "seen_avg": 52.9,
            "unseen_avg": 56.0,
            "seen_step": 30.8,
            "seen_prefill_k": 1.21,
            "unseen_prefill_k": 1.23,
        },
        "Vanilla": {"seen_avg": 43.6, "unseen_avg": 47.0},
    }


def table2_search_qa() -> dict[str, Any]:
    return {
        "LatentSkill": {
            "avg": 35.6,
            "cost_k": 0.31,
            "NQ": 36.2,
            "HotpotQA": 39.6,
            "MuSiQue": 9.8,
        },
        "In-Context Skill": {"avg": 32.6, "cost_k": 1.10, "NQ": 27.2, "HotpotQA": 30.2, "MuSiQue": 7.6},
        "RAG": {"avg": 34.4, "cost_k": 0.89},
    }


def table3_composition_look() -> dict[str, dict[str, float]]:
    return {
        "Look-Only": {"seen": 61.5, "unseen": 72.2},
        "Pick-Only": {"seen": 61.5, "unseen": 61.1},
        "Direct Merging": {"seen": 69.2, "unseen": 61.1},
        "Text Merging": {"seen": 61.5, "unseen": 61.1},
        "Component Merging": {"seen": 84.6, "unseen": 77.8},
    }


def table4_sensitivity() -> dict[str, dict[str, float]]:
    return {
        "Base": {"in_context_alf": 52.9, "latent_alf": 74.3, "in_context_search": 32.6, "latent_search": 35.6},
        "Paraphrase": {"in_context_alf": 50.7, "latent_alf": 67.9, "in_context_search": 33.2, "latent_search": 34.0},
        "Plaintext": {"in_context_alf": 50.0, "latent_alf": 74.3, "in_context_search": 32.4, "latent_search": 34.4},
        "Reorder": {"in_context_alf": 50.7, "latent_alf": 69.3, "in_context_search": 32.4, "latent_search": 33.6},
        "Noise": {"in_context_alf": 47.9, "latent_alf": 71.4, "in_context_search": 31.7, "latent_search": 33.7},
        "Hijack": {"in_context_alf": 8.57, "latent_alf": 38.6, "in_context_search": 23.5, "latent_search": 34.0},
        "Extract": {"in_context_alf": 48.6, "latent_alf": 70.0, "in_context_search": 21.3, "latent_search": 29.3},
    }


def table7_injection_ablation() -> dict[str, dict[str, float]]:
    return {
        "full": {"seen": 63.6, "unseen": 61.2},
        "full:o+d": {"seen": 59.3, "unseen": 63.4},
        "last6": {"seen": 47.1, "unseen": 50.0},
        "first30:o+d": {"seen": 59.3, "unseen": 61.2},
    }


def table8_low_rank_stats() -> dict[str, Any]:
    return {
        "stable_rank_pretrain": {"alfworld": 2.36, "search": 2.40},
        "stable_rank_sft": {"alfworld": 2.18, "search": 2.23},
        "random_lora_stable_rank": 837.87,
        "rank1_energy_pct_sft": 54.2,
        "rank5_energy_pct_sft": 93.7,
    }


def table12_per_task_hijack() -> dict[str, dict[str, float]]:
    """Appendix I per-task Hijack attack (ALFWorld task types, stub anchors)."""
    return {
        "Pick": {"in_context": 7.1, "latent": 41.2},
        "Look": {"in_context": 9.5, "latent": 36.8},
        "Clean": {"in_context": 8.0, "latent": 39.1},
        "Heat": {"in_context": 10.2, "latent": 37.5},
        "Cool": {"in_context": 8.9, "latent": 38.4},
    }


def table11_alpha_sweep() -> dict[str, Any]:
    return {
        "seen_avg_by_alpha": {
            0.0: 43.57,
            0.3: 62.86,
            0.6: 74.29,
            1.0: 63.57,
            1.2: 22.86,
        },
        "unseen_avg_by_alpha": {
            0.0: 47.01,
            0.5: 70.90,
            0.6: 69.40,
            1.0: 61.19,
            1.2: 8.21,
        },
        "optimal_alpha_seen": 0.6,
        "optimal_alpha_unseen": 0.5,
    }


def benchmarks_bundle() -> dict[str, Any]:
    t1 = table1_alfworld()
    t2 = table2_search_qa()
    return {
        "table1_alfworld": t1,
        "table2_search_qa": t2,
        "table3_composition": table3_composition_look(),
        "table4_sensitivity": table4_sensitivity(),
        "table7_injection": table7_injection_ablation(),
        "table8_low_rank": table8_low_rank_stats(),
        "table11_alpha": table11_alpha_sweep(),
        "table12_per_task_hijack": table12_per_task_hijack(),
        "alfworld_gain_seen_pp": t1["LatentSkill"]["seen_avg"] - t1["In-Context Skill"]["seen_avg"],
        "alfworld_gain_unseen_pp": t1["LatentSkill"]["unseen_avg"] - t1["In-Context Skill"]["unseen_avg"],
        "search_gain_pp": t2["LatentSkill"]["avg"] - t2["In-Context Skill"]["avg"],
    }
