"""Paper Tables 1–2, 7 (reported metrics ×100 unless noted)."""

from __future__ import annotations

from typing import Any


def table1_spearman_large() -> list[dict[str, Any]]:
    """Spearman rank correlation ×100: LLM vs human per attribute (Table 1)."""
    rows = [
        # Llama-70B vanilla
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "respect", "rho_x100": -73.03},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "sentiment", "rho_x100": -70.47},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "status", "rho_x100": -57.35},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "hatespeech", "rho_x100": -48.79},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "genocide", "rho_x100": 54.87},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "dehumanize", "rho_x100": 59.78},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "attack_defend", "rho_x100": 61.76},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "violence", "rho_x100": 64.45},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "humiliate", "rho_x100": 66.42},
        {"model": "Llama-70B", "condition": "vanilla", "attribute": "insult", "rho_x100": 69.75},
        # Qwen-72B vanilla (subset for cross-model check)
        {"model": "Qwen-72B", "condition": "vanilla", "attribute": "respect", "rho_x100": -74.44},
        {"model": "Qwen-72B", "condition": "vanilla", "attribute": "insult", "rho_x100": 71.17},
        {"model": "Qwen-72B", "condition": "vanilla", "attribute": "hatespeech", "rho_x100": -51.34},
    ]
    return rows


def table2_ridge_reconstruction() -> list[dict[str, Any]]:
    """Hate score reconstruction R² and classification (Table 2, ×100)."""
    return [
        {
            "model": "Llama-70B",
            "method": "ridge_vanilla",
            "r2_x100": 70.57,
            "r2_std_x100": 0.59,
            "f1_x100": 68.63,
            "acc_x100": 84.02,
            "prec_x100": 71.15,
            "recall_x100": 66.29,
        },
        {
            "model": "Llama-70B",
            "method": "ridge_persona",
            "r2_x100": 70.71,
            "r2_std_x100": 0.45,
            "f1_x100": 68.95,
            "acc_x100": 84.49,
            "prec_x100": 73.07,
            "recall_x100": 65.27,
        },
        {
            "model": "Llama-70B",
            "method": "zero_shot_baseline",
            "r2_x100": None,
            "f1_x100": 60.30,
            "acc_x100": 66.64,
            "prec_x100": 43.95,
            "recall_x100": 96.02,
        },
        {
            "model": "Qwen-72B",
            "method": "ridge_vanilla",
            "r2_x100": 68.85,
            "r2_std_x100": 0.56,
            "f1_x100": 69.20,
            "acc_x100": 83.95,
            "prec_x100": 70.08,
            "recall_x100": 68.34,
        },
        {
            "model": "Qwen-72B",
            "method": "ridge_persona",
            "r2_x100": 68.65,
            "r2_std_x100": 0.60,
            "f1_x100": 67.63,
            "acc_x100": 83.83,
            "prec_x100": 71.67,
            "recall_x100": 64.03,
        },
    ]


def table7_ablation() -> list[dict[str, Any]]:
    """Formula variants for score reconstruction (Table 7, R² ×100)."""
    return [
        {"model": "Llama-70B", "formula": "A_no_ridge_conf_spearman", "r2_x100": -8.44},
        {"model": "Llama-70B", "formula": "B_ridge_confidence", "r2_x100": 70.57},
        {"model": "Llama-70B", "formula": "C_ridge_no_confidence", "r2_x100": 68.63},
        {"model": "Llama-70B", "formula": "D_no_ridge_spearman_only", "r2_x100": -13.24},
        {"model": "Qwen-72B", "formula": "B_ridge_confidence", "r2_x100": 68.85},
        {"model": "Qwen-72B", "formula": "C_ridge_no_confidence", "r2_x100": 68.27},
    ]


def attribute_clusters_from_table1() -> dict[str, str]:
    """Classify attributes by sign of Llama-70B vanilla Spearman."""
    out: dict[str, str] = {}
    for row in table1_spearman_large():
        if row["model"] != "Llama-70B" or row["condition"] != "vanilla":
            continue
        rho = row["rho_x100"]
        out[row["attribute"]] = "behavioral" if rho > 0 else "evaluative"
    return out


def headline_results() -> dict[str, str]:
    return {
        "alignment_split": (
            "Behavioral attributes positively correlated; evaluative attributes inverted "
            "(all four models)."
        ),
        "persona_effect": "Reduces confidence; does not improve Spearman alignment.",
        "reconstruction": "Confidence-weighted Ridge R² up to 70.71 (Llama-70B persona).",
        "vs_baselines": "Ridge pipeline ~84% accuracy vs 57–70% direct prompting; better precision.",
    }
