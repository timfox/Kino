"""Framework card, paper tables, and demos (arXiv:2606.15346)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig
from ltx_trainer.dyna_pruner.layout import LIMITATIONS
from ltx_trainer.dyna_pruner.loss import mse_task_loss, total_loss
from ltx_trainer.dyna_pruner.masks import (
    apply_data_mask,
    importance_from_temporal_variance,
    mask_summary,
)
from ltx_trainer.dyna_pruner.mock import evaluation_smoke, toy_weatherbench_frame
from ltx_trainer.dyna_pruner.synergy import synchronized_masks


def framework_card(cfg: DynaPrunerConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DynaPrunerConfig()
    return {
        "name": "Dyna-Pruner",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "authors": list(cfg.authors),
        "idea": (
            "Input-adaptive co-pruning: shared importance field S yields coupled data mask "
            "M_data and model mask M_weight for per-sample sparse sub-networks at inference."
        ),
        "mechanism": [
            "Mask generator g_mask(X; φ) → importance map S",
            "Data: soft X⊙S (train), hard M_data via STE (inference)",
            "Model: receptive-field aggregate I_k, prune filters below learnable τ",
            "Loss: L_task + λd||S||_1 + λw||I||_1",
        ],
        "defaults": {
            "sd": cfg.data_sparsity_sd,
            "sw": cfg.model_sparsity_sw,
            "lambda_d": cfg.lambda_d,
            "lambda_w": cfg.lambda_w,
        },
        "backbones": list(cfg.backbones),
        "datasets": list(cfg.datasets),
        "baselines": list(cfg.baselines),
        "reported": {
            "gflops_reduction_pct": cfg.typical_gflops_reduction_pct,
            "latency_reduction_pct": cfg.typical_latency_reduction_pct,
            "mse_increase_pct": cfg.typical_mse_increase_pct,
        },
        "limitations": list(LIMITATIONS),
    }


def table_main_results() -> dict[str, dict[str, dict[str, float | str]]]:
    """Table I — WeatherBench / SEVIR / TaxiBJ (SimVP, ConvLSTM, TAU)."""
    return {
        "SimVP": {
            "Dense": {"mse_wb": 0.0452, "mse_sevir": 24.51, "mse_taxibj": 2.51, "gflops": 120.4, "latency_ms": 85.2},
            "MP": {"mse_wb": 0.0471, "mse_sevir": 25.89, "mse_taxibj": 2.63, "gflops": 31.5, "latency_ms": 42.1},
            "Dyna-Pruner": {"mse_wb": 0.0458, "mse_sevir": 24.83, "mse_taxibj": 2.55, "gflops": 30.2, "latency_ms": 33.8},
        },
        "ConvLSTM": {
            "Dense": {"mse_wb": 0.0515, "mse_sevir": 28.14, "mse_taxibj": 2.85, "gflops": 155.8, "latency_ms": 110.5},
            "MP": {"mse_wb": 0.0548, "mse_sevir": 30.02, "mse_taxibj": 3.03, "gflops": 40.1, "latency_ms": 58.3},
            "Dyna-Pruner": {"mse_wb": 0.0523, "mse_sevir": 28.59, "mse_taxibj": 2.89, "gflops": 39.5, "latency_ms": 41.7},
        },
        "TAU": {
            "Dense": {"mse_wb": 0.0439, "mse_sevir": 23.98, "mse_taxibj": 2.42, "gflops": 180.2, "latency_ms": 135.8},
            "MP": {"mse_wb": 0.0465, "mse_sevir": 25.51, "mse_taxibj": 2.58, "gflops": 46.2, "latency_ms": 75.1},
            "Dyna-Pruner": {"mse_wb": 0.0447, "mse_sevir": 24.37, "mse_taxibj": 2.46, "gflops": 45.5, "latency_ms": 50.2},
        },
    }


def table_robustness() -> dict[str, dict[str, float | str]]:
    """Table II — noise and mask-failure robustness (WeatherBench MSE)."""
    return {
        "Dense": {"clean": 0.0452, "noise_0.05": 0.0512, "mask_fail_10pct": "—"},
        "Random_Pruning_70pct": {"clean": 0.0471, "noise_0.05": 0.0584, "mask_fail_10pct": 0.0562},
        "Dyna-Pruner": {"clean": 0.0458, "noise_0.05": 0.0489, "mask_fail_10pct": 0.0491},
    }


def table_sparsity_sensitivity() -> list[dict[str, float | str]]:
    """Table III — sd/sw trade-off (SimVP, excerpt)."""
    return [
        {"sd": 0.5, "sw": 0.5, "mse_wb": 0.0454, "gflops": 58.1, "latency_ms": 55.3},
        {"sd": 0.7, "sw": 0.5, "mse_wb": 0.0456, "gflops": 45.9, "latency_ms": 48.1},
        {"sd": 0.7, "sw": 0.7, "mse_wb": 0.0458, "gflops": 30.2, "latency_ms": 33.8},
        {"sd": 0.9, "sw": 0.7, "mse_wb": 0.0469, "gflops": 21.5, "latency_ms": 28.4},
        {"sd": 0.9, "sw": 0.9, "mse_wb": 0.0482, "gflops": 12.3, "latency_ms": 21.7},
    ]


def table_ablation() -> list[dict[str, float | str | bool]]:
    """Table IV — synergy ablation (WeatherBench)."""
    return [
        {"DP": False, "MP": False, "synergy": False, "mse": 0.0452, "gflops": 120.4, "latency_ms": 85.2},
        {"DP": True, "MP": False, "synergy": False, "mse": 0.0465, "gflops": 65.1, "latency_ms": 58.7},
        {"DP": False, "MP": True, "synergy": False, "mse": 0.0471, "gflops": 31.5, "latency_ms": 42.1},
        {"DP": True, "MP": True, "synergy": False, "mse": 0.0469, "gflops": 30.8, "latency_ms": 35.5},
        {"DP": True, "MP": True, "synergy": True, "mse": 0.0458, "gflops": 30.2, "latency_ms": 33.8},
    ]


def pipeline_demo(cfg: DynaPrunerConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DynaPrunerConfig()
    x = toy_weatherbench_frame()
    s = importance_from_temporal_variance(x)
    sync = synchronized_masks(
        s,
        sd=cfg.data_sparsity_sd,
        sw=cfg.model_sparsity_sw,
        ste_threshold=cfg.ste_threshold,
        kernel=cfg.receptive_field,
    )
    x_sparse = apply_data_mask(x, sync["M_data"])
    l_task = mse_task_loss(x_sparse[-1], x[-1])
    losses = total_loss(
        l_task,
        sync["S"],
        sync["I"],
        lambda_d=cfg.lambda_d,
        lambda_w=cfg.lambda_w,
    )
    return {
        "input_shape": list(x.shape),
        "mask_summary": mask_summary(s, threshold=sync["data_threshold"]),
        "model_keep_ratio": float(sync["M_weight"].mean()),
        "losses": losses,
        "tau": float(sync["tau"]),
    }


def evaluation_demo() -> dict[str, Any]:
    cfg = DynaPrunerConfig()
    dense_gflops = 120.4
    pruned_gflops = 30.2
    return {
        "smoke": evaluation_smoke(),
        "framework": framework_card(cfg),
        "gflops_reduction_pct": round(100.0 * (1.0 - pruned_gflops / dense_gflops), 1),
        "speedup_vs_dense_latency": round(85.2 / 33.8, 2),
        "table_main": table_main_results(),
        "table_robustness": table_robustness(),
        "table_sparsity": table_sparsity_sensitivity(),
        "table_ablation": table_ablation(),
    }
