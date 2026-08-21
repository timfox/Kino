"""Four-dimensional autoscaling taxonomy (Sec. IV, Fig. 7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ltx_trainer.predictive_autoscaling.constants import (
    TAXONOMY_EVALUATION,
    TAXONOMY_PREDICTION_MODELS,
    TAXONOMY_SCALING_TARGETS,
    TAXONOMY_SCALING_TRIGGERS,
)


class ScalingStrategy(StrEnum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    HYBRID = "hybrid"


def taxonomy_card() -> dict[str, Any]:
    return {
        "dimensions": {
            "scaling_triggers": list(TAXONOMY_SCALING_TRIGGERS),
            "scaling_targets": list(TAXONOMY_SCALING_TARGETS),
            "prediction_models": list(TAXONOMY_PREDICTION_MODELS),
            "evaluation": list(TAXONOMY_EVALUATION),
        },
        "strategies": [s.value for s in ScalingStrategy],
        "kubernetes_native": ["HPA", "VPA", "Cluster Autoscaler", "KEDA", "CRD", "Operator"],
        "federated": ["KubeFlower", "FedALoRA", "FedInv", "DP-aware scaling"],
        "drift_aware": ["ADI", "FRSC", "RRS", "CDT", "uncertainty correction"],
    }


def classify_approach(name: str) -> dict[str, str]:
    """Route a named system/paper stub to taxonomy axes."""
    key = name.lower().replace("-", " ").strip()
    mapping: dict[str, dict[str, str]] = {
        "hpa": {"target": "hpa", "strategy": "reactive", "trigger": "threshold"},
        "keda": {"target": "keda", "strategy": "reactive", "trigger": "event_driven"},
        "informerautoscale": {"target": "hpa", "strategy": "proactive", "model": "transformer"},
        "catscaler": {"target": "hpa", "strategy": "proactive", "model": "transformer"},
        "mv-transformer mape": {"target": "hpa", "strategy": "proactive", "model": "mv_transformer"},
        "kubeflower": {"target": "hpa", "strategy": "proactive", "model": "deep_learning"},
        "predictiveautoscaler crd": {"target": "hpa", "strategy": "proactive", "model": "transformer"},
    }
    for prefix, tags in mapping.items():
        if prefix in key or key in prefix:
            return {"name": name, **tags}
    return {
        "name": name,
        "target": "hpa",
        "strategy": "hybrid",
        "trigger": "hybrid",
        "model": "machine_learning",
    }
