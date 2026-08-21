"""Paper benchmark anchors and capability tables (Section 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dopr.config import BenchmarkAnchors


def table_gsm8k_sft_3b() -> list[dict[str, Any]]:
    return [
        {"optimizer": "AdamW", "peak_gsm8k_pct": 72.0, "final_train_loss_rel": 1.0},
        {"optimizer": "DoPr-AdamW", "peak_gsm8k_pct": 76.5, "final_train_loss_rel": 1.05},
        {"optimizer": "Muon", "peak_gsm8k_pct": 70.0, "final_train_loss_rel": 1.0},
        {"optimizer": "DoPr-Muon", "peak_gsm8k_pct": 74.0, "final_train_loss_rel": 1.08},
    ]


def table_gsm8k_sft_8b_lr_sweep() -> list[dict[str, Any]]:
    lrs = ["2e-5", "5e-5", "7e-5", "1e-4", "2e-4", "5e-4", "7e-4"]
    return [
        {
            "learning_rate": lr,
            "adamw_gsm8k": {"2e-5": 71.4, "5e-4": 66.2, "7e-4": 56.0}.get(lr, 68.0),
            "dopr_adamw_gsm8k": {"5e-4": 80.5, "7e-4": 70.4, "2e-5": 71.9}.get(lr, 70.0),
        }
        for lr in lrs
    ]


def table_humanoid_terminal_reward(anchors: BenchmarkAnchors | None = None) -> list[dict[str, Any]]:
    a = anchors or BenchmarkAnchors()
    return [
        {"gp": "AdamW", "median_return": a.humanoid_adamw_median, "dopr_median_return": a.humanoid_dopr_adamw_median},
        {"gp": "Muon", "median_return": a.humanoid_adamw_median * 0.98, "dopr_median_return": a.humanoid_dopr_adamw_median * 1.02},
        {"gp": "Signum", "median_return": a.humanoid_adamw_median * 0.97, "dopr_median_return": a.humanoid_dopr_adamw_median * 0.99},
        {"gp": "AdaMuon", "median_return": a.humanoid_adamw_median * 0.99, "dopr_median_return": a.humanoid_dopr_adamw_median * 1.01},
    ]


def table_robomimic_success(anchors: BenchmarkAnchors | None = None) -> list[dict[str, Any]]:
    a = anchors or BenchmarkAnchors()
    return [
        {
            "task": "tool_hang_ph",
            "adamw": a.tool_hang_adamw_best,
            "dopr_adamw": a.tool_hang_dopr_adamw_best,
            "muon": a.tool_hang_adamw_best * 0.95,
            "dopr_muon": a.tool_hang_dopr_adamw_best * 1.02,
        },
        {
            "task": "transport_ph",
            "adamw": a.transport_adamw_best,
            "dopr_adamw": a.transport_dopr_adamw_best,
            "muon": a.transport_adamw_best * 1.03,
            "dopr_muon": a.transport_dopr_adamw_best * 0.98,
        },
    ]


def table_sit_fid_imagenet256() -> list[dict[str, Any]]:
    return [
        {"optimizer": "AdamW", "fid_50k_final": 12.5, "steps_k": 400},
        {"optimizer": "DoPr-AdamW", "fid_50k_final": 11.2, "steps_k": 400},
        {"optimizer": "Muon", "fid_50k_final": 11.8, "steps_k": 200},
        {"optimizer": "DoPr-Muon", "fid_50k_final": 10.9, "steps_k": 200},
    ]


def operating_guidelines() -> list[str]:
    return [
        "Apply AP before GP: M = G Σ_z^{-1}, then D = GP(M).",
        "Use trace-scaled damping γ tr(Σ)/d for scale-invariant Cholesky stability.",
        "Inherit μP LR/WD scaling from the base GP (Observation 4.3).",
        "Prefer low/no dropout when maintaining activation covariance EMA buffers.",
        "Use SUA for Conv2d; full Cholesky for linear / attention QKV projections.",
        "Downstream metrics may improve while validation loss worsens — evaluate rollouts.",
    ]
