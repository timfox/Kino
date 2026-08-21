"""Table anchors and survey comparison stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.constants import (
    LITERATURE_AFTER_DEDUP,
    LITERATURE_EXCLUDED,
    LITERATURE_FINAL,
    LITERATURE_IDENTIFIED,
    LITERATURE_INCLUDED,
    LITERATURE_SCREENED,
)


def literature_funnel() -> dict[str, int]:
    return {
        "identified": LITERATURE_IDENTIFIED,
        "after_dedup": LITERATURE_AFTER_DEDUP,
        "screened": LITERATURE_SCREENED,
        "included": LITERATURE_INCLUDED,
        "excluded": LITERATURE_EXCLUDED,
        "final": LITERATURE_FINAL,
    }


# Table I feature matrix: ✓ full, G# partial, × none
TABLE_I_SURVEY_COMPARISON: list[dict[str, Any]] = [
    {"reference": "Qu et al. (2018)", "reactive": "G#", "proactive": "✓", "hybrid": "✓", "k8s": "G#", "fl": "×", "predict": "G#", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Garí et al. (2021)", "reactive": "×", "proactive": "×", "hybrid": "×", "k8s": "G#", "fl": "×", "predict": "G#", "crd": "×", "drift": "×", "cloud_edge": "G#"},
    {"reference": "Kashyap et al. (2023)", "reactive": "G#", "proactive": "✓", "hybrid": "G#", "k8s": "G#", "fl": "×", "predict": "✓", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Pham and Kim (2024)", "reactive": "G#", "proactive": "✓", "hybrid": "G#", "k8s": "✓", "fl": "✓", "predict": "G#", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Dogani and Khunjush (2024)", "reactive": "G#", "proactive": "✓", "hybrid": "G#", "k8s": "G#", "fl": "✓", "predict": "G#", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Van Do et al. (2025)", "reactive": "✓", "proactive": "×", "hybrid": "G#", "k8s": "✓", "fl": "×", "predict": "×", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Jeong and Jeong (2025)", "reactive": "✓", "proactive": "✓", "hybrid": "✓", "k8s": "✓", "fl": "×", "predict": "G#", "crd": "×", "drift": "×", "cloud_edge": "✓"},
    {"reference": "Meng et al. (2025) CATScaler", "reactive": "G#", "proactive": "✓", "hybrid": "G#", "k8s": "✓", "fl": "×", "predict": "✓", "crd": "×", "drift": "×", "cloud_edge": "G#"},
    {"reference": "Kumar et al. (2026) K8s runtime", "reactive": "×", "proactive": "×", "hybrid": "×", "k8s": "✓", "fl": "×", "predict": "×", "crd": "✓", "drift": "×", "cloud_edge": "✓"},
    {"reference": "This Work", "reactive": "✓", "proactive": "✓", "hybrid": "✓", "k8s": "✓", "fl": "✓", "predict": "✓", "crd": "✓", "drift": "✓", "cloud_edge": "✓"},
]


def table_iii_predictive_models() -> list[dict[str, str]]:
    return [
        {"category": "Informer-Based Forecasting", "use": "long-horizon proactive K8s autoscaling"},
        {"category": "MV-Transformer", "use": "multivariate MAPE cloud-native allocation"},
        {"category": "Deep Learning Forecasting", "use": "LSTM/GRU/CNN-LSTM bursty workloads"},
        {"category": "PredictiveAutoscaler CRD", "use": "declarative forecast policies in control plane"},
        {"category": "Operator Reconciliation", "use": "observe-predict-plan-act loop"},
        {"category": "Model Control System", "use": "adaptive model selection by validation error"},
    ]


def table_v_drift_mechanisms() -> list[dict[str, str]]:
    return [
        {"mechanism": "Autoscaling Drift", "role": "cumulative forecast vs actual mismatch"},
        {"mechanism": "ADI", "role": "round-level drift index for correction trigger"},
        {"mechanism": "Uncertainty correction", "role": "bounded feedback on prediction error"},
        {"mechanism": "RRS", "role": "gradual safe scale-in"},
        {"mechanism": "FRSC", "role": "straggler prediction and minimum CPU guarantees"},
    ]


def table_vi_open_challenges() -> list[dict[str, str]]:
    return [
        {"topic": "Forecasting under drift", "direction": "online learning, hybrid Informer+CNN-LSTM"},
        {"topic": "Multivariate prediction", "direction": "causal feature selection, hierarchical MV-Transformer"},
        {"topic": "CRDs and operators", "direction": "standardised schemas, cross-CRD arbitration"},
        {"topic": "FL heterogeneity", "direction": "DP-cost modelling, privacy-aware scaling"},
        {"topic": "Energy and carbon", "direction": "green-aware forecasting and scheduling"},
    ]
