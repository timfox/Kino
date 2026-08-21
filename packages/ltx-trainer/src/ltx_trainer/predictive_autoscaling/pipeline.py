"""End-to-end predictive autoscaling demo pipeline."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.predictive_autoscaling.config import PredictiveAutoscalingConfig
from ltx_trainer.predictive_autoscaling.drift import drift_demo, resource_removal_strategy
from ltx_trainer.predictive_autoscaling.federated import desired_pods, fl_round_demand
from ltx_trainer.predictive_autoscaling.mape import run_mape_cycle
from ltx_trainer.predictive_autoscaling.models import select_model_stub


def toy_forecast(last_rpm: float, *, horizon: int = 1, jitter: float = 0.05) -> float:
    """Appendix A toy predictor stub."""
    return max(0.0, last_rpm * (1.0 + jitter * (horizon - 1)))


def plan_replicas(
    forecast_rpm: float,
    cfg: PredictiveAutoscalingConfig,
    *,
    current_replicas: int,
) -> dict[str, Any]:
    raw = desired_pods(forecast_rpm, pod_capacity=cfg.pod_capacity_rpm)
    raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))
    after_rrs = resource_removal_strategy(current_replicas, raw, gamma=cfg.rrs_factor)
    return {
        "forecast_rpm": forecast_rpm,
        "desired_raw": raw,
        "desired_after_rrs": after_rrs,
        "current_replicas": current_replicas,
    }


def run_demo(
    *,
    last_rpm: float = 7200.0,
    current_replicas: int = 8,
    observed_cpu: list[float] | None = None,
    predicted_cpu: list[float] | None = None,
    cfg: PredictiveAutoscalingConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PredictiveAutoscalingConfig()
    model = select_model_stub("bursty multivariate")
    forecast = toy_forecast(last_rpm, horizon=cfg.forecast_horizon_min)
    plan = plan_replicas(forecast, cfg, current_replicas=current_replicas)
    mape = run_mape_cycle(
        {"cpu_util": 0.72, "memory_util": 0.61, "rps": last_rpm / 60.0},
        forecast_demand=forecast,
        pod_capacity=cfg.pod_capacity_rpm,
        current_replicas=current_replicas,
        min_replicas=cfg.min_replicas,
        max_replicas=cfg.max_replicas,
    )
    fl_cpu = fl_round_demand(c_train=0.55, c_comm=0.25, dp_overhead=0.12)
    obs = observed_cpu or [0.7, 0.75, 0.82, 0.78]
    pred = predicted_cpu or [0.68, 0.72, 0.74, 0.76]
    drift = drift_demo(obs, pred, tau_adi=cfg.adi_threshold)
    return {
        "model_selected": model,
        "mape": mape,
        "scaling_plan": plan,
        "fl_cpu_demand": fl_cpu,
        "fl_pods": desired_pods(max(forecast, fl_cpu * cfg.pod_capacity_rpm), pod_capacity=cfg.pod_capacity_rpm),
        "drift": drift,
    }


def replica_from_forecast(forecast: float, pod_capacity: float) -> int:
    return max(1, int(math.ceil(forecast / max(pod_capacity, 1.0))))
