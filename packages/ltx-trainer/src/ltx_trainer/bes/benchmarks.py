"""Paper benchmarks — Tables 1–4, Figure 3 anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bes.constants import (
    MUSIQUE_POST_TRAIN,
    MUSIQUE_WALLTIME_SEC,
    OPEN_PROBLEM_API_COST_USD,
    OPEN_PROBLEM_RESULTS,
)


def table1_musique() -> dict[str, Any]:
    """Table 1 — MuSiQue post-training."""
    return {"rows": MUSIQUE_POST_TRAIN, "metric": "accuracy_pct"}


def table2_open_problems() -> dict[str, Any]:
    """Table 2 — GPT-5 open problem solving (SkyDiscover setting)."""
    return OPEN_PROBLEM_RESULTS


def table3_post_train_cost() -> dict[str, Any]:
    """Table 3 — wall-clock per training step (3B MuSiQue)."""
    return {
        "walltime_sec": MUSIQUE_WALLTIME_SEC,
        "bes_accuracy": MUSIQUE_POST_TRAIN["Llama-3.2-3B-Instruct"]["BES_acc"],
        "tree_grpo_accuracy": MUSIQUE_POST_TRAIN["Llama-3.2-3B-Instruct"]["Tree-GRPO_acc"],
    }


def table4_inference_cost() -> dict[str, Any]:
    """Table 4 — average API cost per generation."""
    return OPEN_PROBLEM_API_COST_USD


def figure3_kk_trend() -> dict[str, float]:
    """Figure 3 — log(accuracy) EMA anchors (Knights-and-Knaves validation)."""
    return {
        "GRPO_end_log_acc": 2.55,
        "MaxRL_end_log_acc": 2.58,
        "BES_end_log_acc": 2.95,
        "note": "BES steadily improves where GRPO/MaxRL plateau on hard K&K set",
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_musique": table1_musique(),
        "table2_open_problems": table2_open_problems(),
        "table3_post_train_cost": table3_post_train_cost(),
        "table4_inference_cost": table4_inference_cost(),
        "figure3_kk": figure3_kk_trend(),
        "paper": "arXiv:2605.28814",
        "code": "https://github.com/Embodied-Minds-Lab/BES",
    }
