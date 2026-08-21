"""Framework card, demos, and benchmark manifest (arXiv:2605.24322)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.intphys import synthetic_intphys_features
from ltx_trainer.physics_steering.ltx_bridge import PhysicsSteeringLTXBridge, ltx_integration_notes
from ltx_trainer.physics_steering.metrics import directional_purity, flip_rate, score_delta
from ltx_trainer.physics_steering.paper_tables import (
    layer_accuracy_dict,
    table_alpha_sweep,
    table_block_cav_disentanglement,
    table_layer_ablation,
    table_probe_accuracy_by_layer,
    table_subspace_orthogonality,
)
from ltx_trainer.physics_steering.pez import identify_pez_layers, top_pez_layers
from ltx_trainer.physics_steering.probe import cav_from_weights, fit_probe_with_pca, fit_logistic_probe
from ltx_trainer.physics_steering.representation import batch_mean_pool, mean_pool_hidden
from ltx_trainer.physics_steering.experiments import run_synthetic_paper_benchmark
from ltx_trainer.physics_steering.steering import (
    angle_between,
    iterative_orthogonal_probe_accuracies,
    steer_batch_scores,
    steer_hidden_states,
)


def framework_card(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysicsSteeringConfig()
    acc = layer_accuracy_dict()
    return {
        "name": "Physics Steering",
        "paper": cfg.paper_arxiv,
        "title": "Causal Physics Steering in Video World Models via Concept Activation Vectors",
        "author": "Nahid Alam (Oreon Labs / Cohere Labs Community)",
        "backbone": cfg.model_id,
        "method": "Training-free CAV injection at PEZ layers (Eq. 5)",
        "primary_pez_layer": cfg.primary_pez_layer,
        "top_pez_layers": list(cfg.top_pez_layers),
        "steering_saturation_alpha": cfg.steering_saturation_alpha,
        "reported_effects": {
            "directional_purity_at_pez": 1.0,
            "cosine_shift_at_alpha5": 0.775,
            "physics_vs_motion_angle_deg": 90.0,
        },
        "pez_layers_from_paper": top_pez_layers(acc, epsilon=cfg.pez_epsilon),
        "builds_on": "Joseph et al. PEZ analysis on VideoMAE; IntPhys benchmark",
    }


def paper_limitations() -> list[str]:
    return [
        "VideoMAE encoder only — no pixel-space steered video without MAE decoder coupling.",
        "IntPhys dev split used for balanced probe training (official train split lacks violations).",
        "Rigid single-vector steering; full 2–3D physics subspace not yet composed with motion.",
        "PEZ localization validated on VideoMAE-base; other architectures require re-probing.",
    ]


def benchmark_manifest(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysicsSteeringConfig()
    return {
        "dataset": "IntPhys dev split (stratified 60/20/20)",
        "blocks": list(cfg.intphys_blocks),
        "metrics": ["probe_accuracy", "flip_rate", "score_delta", "directional_purity", "representation_drift"],
        "hardware_note": "NVIDIA L40S; activation collection ~3–5 min/split",
        "mask_pipeline": None,
        "model": cfg.model_id,
    }


def training_step_demo(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysicsSteeringConfig()
    torch.manual_seed(11)
    dim = min(cfg.hidden_dim, 32)
    x, y, _ = synthetic_intphys_features(64, dim, seed=3)
    w, b, acc = fit_logistic_probe(x, y, steps=cfg.probe_train_steps)
    v = cav_from_weights(w)
    acc_dict = {i: 0.65 + 0.05 * (5 - abs(i - 5)) / 5 for i in range(cfg.num_layers)}
    pez = identify_pez_layers(acc_dict, epsilon=cfg.pez_epsilon)
    return {
        "probe_accuracy": round(acc, 4),
        "cav_norm": round(float(v.norm().item()), 4),
        "pez_layers_smoke": pez[:6],
        "primary_pez": cfg.primary_pez_layer,
        "sign_convention": "alpha>0 → impossible; alpha<0 → possible",
    }


def evaluation_demo(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysicsSteeringConfig()
    torch.manual_seed(19)
    dim = min(cfg.hidden_dim, 48)
    n = 48
    x, y, blocks = synthetic_intphys_features(n, dim, seed=5)
    w, b, acc = fit_probe_with_pca(x, y, n_components=min(cfg.pca_components, n - 1), steps=cfg.probe_train_steps)
    v = cav_from_weights(w)

    base_p, steered_p_pos, pred_pos = steer_batch_scores(x, w, b, v, alpha=5.0)
    _, steered_p_neg, pred_neg = steer_batch_scores(x, w, b, v, alpha=-5.0)
    base_pred = (base_p >= 0.5).long()
    fr_pos = flip_rate(base_pred, pred_pos)
    fr_neg = flip_rate(base_pred, pred_neg)

    H = torch.randn(8, dim)
    H_steered = steer_hidden_states(H, v, alpha=5.0)
    f0 = mean_pool_hidden(H)
    f1 = mean_pool_hidden(H_steered)
    dp = directional_purity(f1 - f0, v)

    orth = iterative_orthogonal_probe_accuracies(x, y, max_iters=3, fit_steps=100)
    motion_dir = torch.randn(dim)
    motion_dir[0] = 0.0
    physics_motion_angle = angle_between(v, motion_dir)

    bridge = PhysicsSteeringLTXBridge(cfg, dim=dim)
    bridge_stats = bridge.fit_from_labels(x, y, steps=100)
    ltx_out = bridge(x[:4], alpha=5.0)

    alpha_tab = table_alpha_sweep()
    ablation = table_layer_ablation()
    post_pez_flip_zero = all(r["flip_rate"] == 0.0 for r in ablation if r["injection_layer"] >= 6)

    return {
        **training_step_demo(cfg),
        "pca_probe_accuracy": round(acc, 4),
        "flip_rate_alpha_pos5": round(fr_pos, 4),
        "flip_rate_alpha_neg5": round(fr_neg, 4),
        "score_delta_alpha_pos5": round(score_delta(base_p, steered_p_pos), 4),
        "directional_purity_hidden": round(dp, 4),
        "orthogonal_probe_accuracies": [round(a, 4) for a in orth],
        "physics_vs_motion_angle_deg_smoke": round(physics_motion_angle, 2),
        "ltx_bridge_scores_shape": list(ltx_out.shape),
        "ltx_bridge_fit": {k: round(v, 4) if isinstance(v, float) else v for k, v in bridge_stats.items()},
        "paper_alpha5_p_impossible": next(r["p_impossible"] for r in alpha_tab if r["alpha"] == 5),
        "paper_layer5_dp": next(r["directional_purity"] for r in ablation if r["injection_layer"] == 5),
        "post_pez_layers_flip_zero": post_pez_flip_zero,
        "block_cav": table_block_cav_disentanglement(),
        "subspace_orthogonality": table_subspace_orthogonality(),
        "batch_pool_shape": list(batch_mean_pool(torch.randn(2, 10, dim)).shape),
        "conclusion": "CAV steering shifts P(impossible) bidirectionally; PEZ-localized effect in paper tables.",
        "synthetic_benchmark": run_synthetic_paper_benchmark(cfg),
    }
