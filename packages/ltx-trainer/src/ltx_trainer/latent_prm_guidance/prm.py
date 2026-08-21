"""Latent process reward model (§2.2–2.4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.constants import (
    BRANCHES_TEST,
    BRANCHES_TRAIN,
    GENERATOR_HIDDEN_DIM,
    LATENT_STEPS_INFER,
    LATENT_STEPS_TRAIN,
    PRM_BACKBONE,
    PRM_HIDDEN_DIM,
)


def prm_architecture_card() -> dict[str, Any]:
    return {
        "backbone": PRM_BACKBONE,
        "head": "scalar value Vθ(τ≤t)",
        "adapter": f"linear+GELU+LayerNorm {GENERATOR_HIDDEN_DIM}→{PRM_HIDDEN_DIM}",
        "input": "prefix trajectory τ≤t = (h1, …, ht) from frozen primary",
        "target": "mean terminal reward Rt over B rollout continuations",
        "loss": "MSE(Vθ(τ≤t), Rt)",
    }


def training_card() -> dict[str, Any]:
    return {
        "latent_steps_collect": LATENT_STEPS_TRAIN,
        "branches_per_step": BRANCHES_TRAIN,
        "branch_composition": "2 perturbed + 1 unperturbed",
        "optimizer": "AdamW two-stage (adapter+head, then full)",
        "lr": 1e-6,
        "effective_batch_size": 6,
        "dev_samples": 60,
        "final_train_trajectories": 952,
    }


def inference_card() -> dict[str, Any]:
    return {
        "latent_steps": LATENT_STEPS_INFER,
        "branches_per_step": BRANCHES_TEST,
        "branch_composition": "1 unperturbed + 7 perturbed",
        "selection": "greedy argmax Vθ at each step; single retained branch",
        "after_latent": "standard autoregressive code decoding",
    }


def prefix_target_mean(terminal_scores: list[float]) -> float:
    """Rt = (1/B) Σ_b S(y_t^(b))."""
    if not terminal_scores:
        raise ValueError("terminal_scores must be non-empty")
    return sum(terminal_scores) / len(terminal_scores)


def mse_loss(predicted: float, target: float) -> float:
    return (predicted - target) ** 2
