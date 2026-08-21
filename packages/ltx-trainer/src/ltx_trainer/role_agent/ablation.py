"""Table 3/4 ablations and sensitivity helpers (CPU toy)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.benchmarks import table3_ablation_qwen15b, table4_hyperparam_sensitivity
from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.training import run_toy_training


def ablation_table_anchors() -> dict[str, Any]:
    """Paper Table 3 static anchors."""
    return table3_ablation_qwen15b()


def sensitivity_table_anchors() -> dict[str, Any]:
    """Paper Table 4 static anchors."""
    return table4_hyperparam_sensitivity()


def run_ablation_smoke(*, iterations: int = 8, seed: int = 0) -> dict[str, Any]:
    """CPU toy runs for full / w/o WIA / w/o AIW (mirrors Table 3 toggles)."""
    full = run_toy_training(iterations=iterations, seed=seed, enable_wia=True, enable_aiw=True)
    no_aiw = run_toy_training(iterations=iterations, seed=seed, enable_wia=True, enable_aiw=False)
    no_wia = run_toy_training(iterations=iterations, seed=seed, enable_wia=False, enable_aiw=True)
    anchors = ablation_table_anchors()
    return {
        "ok": full["ok"] and no_aiw["ok"] and no_wia["ok"],
        "paper_anchors": anchors,
        "toy_full": {"final_sr": full["final_success_rate"], "modes": len(full["failure_mode_evolution"])},
        "toy_wo_aiw": {"final_sr": no_aiw["final_success_rate"], "modes": len(no_aiw["failure_mode_evolution"])},
        "toy_wo_wia": {"final_sr": no_wia["final_success_rate"], "modes": len(no_wia["failure_mode_evolution"])},
        "note": "Toy env success rates are smoke-only; paper Table 3 numbers are upstream VeRL anchors.",
    }


def sensitivity_configs() -> list[RoleAgentConfig]:
    """Table 4 α and H grid (config objects only)."""
    base = RoleAgentConfig()
    return [
        RoleAgentConfig(advantage_alpha=0.5),
        base,
        RoleAgentConfig(advantage_alpha=2.0),
        RoleAgentConfig(prediction_horizon_frac=0.05),
        RoleAgentConfig(prediction_horizon_frac=0.10),
        RoleAgentConfig(prediction_horizon_frac=0.20),
    ]
