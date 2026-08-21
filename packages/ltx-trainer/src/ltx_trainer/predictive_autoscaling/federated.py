"""Federated learning autoscaling (Sec. VI)."""

from __future__ import annotations

import math
from typing import Any


def fl_round_demand(
    *,
    c_train: float,
    c_comm: float,
    dp_overhead: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """FL-aware CPU demand max(forecast, CPU_FL) (Fig. 10–11)."""
    cpu_fl = alpha * c_train + beta * c_comm + dp_overhead
    return cpu_fl


def desired_pods(
    cpu_demand: float,
    *,
    pod_capacity: float,
) -> int:
    """P_{t+1} = ceil(D / C_pod) (Eq. 4)."""
    return max(1, int(math.ceil(cpu_demand / max(pod_capacity, 1.0))))


def kubeflower_overview() -> dict[str, Any]:
    return {
        "operator": "KubeFlower",
        "crds": ["PredictiveAutoscaler", "FederatedJob"],
        "isolation": ["containerd", "CRI-O", "Docker sandbox"],
        "multi_region": "AggregateΠ global scaling signal across clusters",
    }


def fl_autoscaling_card() -> dict[str, Any]:
    return {
        "workload_pattern": "discrete FL rounds: broadcast → local train → upload → aggregate",
        "challenges": [
            "client heterogeneity",
            "non-IID data",
            "communication variability",
            "DP overhead",
            "stragglers",
        ],
        "frameworks": ["FedALoRA", "FedInv", "FedScaleEdge", "KubeFlower", "Dogani proactive edge FL"],
        "kubernetes": ["VPA for elastic FL", "operator reconciliation", "container quotas"],
    }
