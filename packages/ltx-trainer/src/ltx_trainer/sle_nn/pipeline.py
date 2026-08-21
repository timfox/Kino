"""Framework card, demos, smoke for SLE κ NN (arXiv:2606.02682)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sle_nn.config import SleNNConfig
from ltx_trainer.sle_nn.loewner_deterministic import inverse_loewner_map, loewner_exponent_a
from ltx_trainer.sle_nn.neural_net import architecture_summary
from ltx_trainer.sle_nn.paper_tables import knowledge_card, simulation_hyperparams, sle_regimes, test_losses
from ltx_trainer.sle_nn.simulation import full_pipeline_demo


def framework_card(cfg: SleNNConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SleNNConfig()
    sim = cfg.sim
    return {
        "name": "SLE-NN",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "title": "Neural networks predict Schramm–Loewner κ from Loewner trajectories",
        "task": "Deterministic c from ξ(t)=c√t; stochastic κ from SLEκ; traces via NV (Sec. 4–5)",
        "pipeline": [
            "Exact inverse Loewner map for ξ(t)=c√t (Eq. 2) + Newton inversion",
            "2×64 ReLU MLP → driver constant c (test MSE ≈ 0.00264)",
            "Euler–Maruyama SLEκ with κ ∈ [0,8] (Eq. 3)",
            "Flatten + 128/64 ReLU + dropout → κ (same/different noise)",
            "Ninomiya–Victoir SLE traces, fixed Brownian (MSE ≈ 0.194)",
        ],
        "simulation": {
            "c_range": [sim.c_min, sim.c_max],
            "kappa_range": [sim.kappa_min, sim.kappa_max],
            "t_grid": [sim.t_start, sim.t_end, sim.n_steps],
            "n_trajectories": sim.n_trajectories,
            "train_fraction": sim.train_fraction,
        },
        "test_losses": test_losses(),
        "physics_anchors": knowledge_card()["physics_anchors"],
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no Keras/TensorFlow training or mpmath high-precision inversion.",
        "SLE simulation uses simplified Euler–Maruyama, not Foster–Lyons–Margarint NV traces.",
        "Reported test MSE values are table anchors with synthetic prediction noise.",
        "No lattice-model interface extraction; κ-from-trace is a fixed-BM proxy only.",
        "Classification κ=κ′ vs κ≠κ′ future direction (Sec. 5) not implemented.",
    ]


def evaluation_demo(cfg: SleNNConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SleNNConfig()
    demo = full_pipeline_demo(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "knowledge": knowledge_card(),
        "test_losses": test_losses(),
        "sle_regimes": sle_regimes(),
        "simulation_hyperparams": simulation_hyperparams(),
        "architecture": architecture_summary(cfg.nn),
        "loewner_a_example": loewner_exponent_a(1.0),
        "inverse_map_example": complex(inverse_loewner_map(0.5 + 0.2j, 0.5, 1.0)),
        "pipeline_demo": demo,
    }


def evaluation_smoke(cfg: SleNNConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    cfg = cfg or SleNNConfig()
    pipe = demo["pipeline_demo"]
    anchors = demo["test_losses"]

    assert 0.0 < demo["loewner_a_example"] < 1.0
    assert pipe["deterministic_test_mse"] < anchors["deterministic_c_mse"] * 3.0
    assert pipe["sle_same_noise_test_mse"] < anchors["sle_kappa_same_noise_mse"] * 2.0
    assert pipe["sle_different_noise_test_mse"] > pipe["sle_same_noise_test_mse"]
    assert pipe["sle_trace_fixed_bm_test_mse"] < anchors["sle_trace_fixed_brownian_mse"] * 2.0
    assert pipe["train_fraction"] == 0.8

    return {
        "status": "ok",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "deterministic_test_mse": pipe["deterministic_test_mse"],
        "sle_same_noise_mse": pipe["sle_same_noise_test_mse"],
        "sle_trace_mse": pipe["sle_trace_fixed_bm_test_mse"],
        "demo_keys": list(demo.keys()),
    }
