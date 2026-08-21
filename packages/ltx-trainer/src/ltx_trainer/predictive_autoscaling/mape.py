"""MAPE control loop helpers (Sec. III-B, Fig. 5–6)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.constants import MAPE_PHASES


def mape_overview() -> dict[str, Any]:
    return {
        "phases": list(MAPE_PHASES),
        "monitor_sources": ["Metrics Server", "Prometheus", "cAdvisor", "custom instrumentation"],
        "analyze_models": ["ARIMA", "LSTM", "Informer", "MV-Transformer", "CNN-LSTM"],
        "plan_outputs": ["desired_replicas", "VPA requests", "node scale-out"],
        "execute_actors": ["HPA", "VPA", "Cluster Autoscaler", "CRD Operator", "KEDA"],
    }


def run_mape_cycle(
    metrics: dict[str, float],
    *,
    forecast_demand: float,
    pod_capacity: float,
    current_replicas: int,
    min_replicas: int = 2,
    max_replicas: int = 50,
) -> dict[str, Any]:
    """Single MAPE iteration: metrics → forecast → replica plan."""
    import math

    desired = int(math.ceil(forecast_demand / max(pod_capacity, 1.0)))
    desired = max(min_replicas, min(max_replicas, desired))
    return {
        "monitor": metrics,
        "analyze": {"forecast_demand": forecast_demand},
        "plan": {"desired_replicas": desired, "delta": desired - current_replicas},
        "execute": {"action": "scale" if desired != current_replicas else "hold"},
    }
