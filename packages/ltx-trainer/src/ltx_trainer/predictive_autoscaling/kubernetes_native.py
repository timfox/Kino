"""Kubernetes autoscaling mechanisms (Sec. III-A, V)."""

from __future__ import annotations

from typing import Any


def k8s_autoscaler_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "HPA",
            "layer": "pod_replicas",
            "strategy": "reactive",
            "metrics": ["CPU", "memory", "custom"],
        },
        {
            "name": "VPA",
            "layer": "pod_resources",
            "strategy": "reactive",
            "metrics": ["historical CPU/memory"],
        },
        {
            "name": "Cluster Autoscaler",
            "layer": "nodes",
            "strategy": "reactive",
            "metrics": ["scheduling pressure", "utilization"],
        },
        {
            "name": "KEDA",
            "layer": "event_driven_pods",
            "strategy": "reactive",
            "metrics": ["queue depth", "Kafka lag", "custom external"],
        },
        {
            "name": "PredictiveAutoscaler CRD",
            "layer": "cr_d_operator",
            "strategy": "proactive",
            "metrics": ["Prometheus multivariate", "forecast horizon"],
        },
    ]


def predictive_autoscaler_crd_example(
    *,
    deployment: str = "web-deploy",
    model_type: str = "informer",
    horizon_min: int = 5,
    pod_capacity: int = 1000,
    rrs: float = 0.6,
    min_replicas: int = 2,
    max_replicas: int = 50,
) -> dict[str, Any]:
    """Appendix A Listing 7 / 10 CR stub."""
    return {
        "apiVersion": "autoscaling.example.com/v1",
        "kind": "PredictiveAutoscaler",
        "metadata": {"name": "web-predictive"},
        "spec": {
            "targetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment},
            "model": {"type": model_type, "horizon": f"{horizon_min}m"},
            "capacity": {"podCapacity": pod_capacity},
            "policy": {"rrs": rrs, "minReplicas": min_replicas, "maxReplicas": max_replicas},
        },
        "status": {"lastForecast": {"value": 8400}, "currentReplicas": 8},
    }


def operator_reconciliation_steps() -> list[str]:
    """Listing 8 operator workflow."""
    return [
        "Watch PredictiveAutoscaler instances",
        "Read spec and current status",
        "Query Prometheus metrics",
        "Invoke model POST /predict",
        "Compute desired replicas ceil(forecast / POD_CAP)",
        "Apply RRS for safe scale-in",
        "Patch Deployment and update CRD status",
    ]
