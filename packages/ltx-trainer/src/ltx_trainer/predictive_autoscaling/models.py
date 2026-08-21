"""Prediction model families for proactive autoscaling (Sec. IV-C, Table II)."""

from __future__ import annotations

from typing import Any


def prediction_model_catalog() -> list[dict[str, Any]]:
    return [
        {
            "family": "statistical",
            "examples": ["ARIMA", "Holt-Winters"],
            "strengths": ["low cost", "interpretable", "seasonal baselines"],
            "limitations": ["linear assumptions", "weak on bursty multivariate"],
        },
        {
            "family": "machine_learning",
            "examples": ["Random Forest", "SVR", "Gradient Boosting"],
            "strengths": ["non-linear multivariate", "moderate data needs"],
            "limitations": ["concept drift", "feature engineering"],
        },
        {
            "family": "deep_learning",
            "examples": ["LSTM", "GRU", "Bi-LSTM", "CNN-LSTM"],
            "strengths": ["temporal dependencies", "proactive accuracy"],
            "limitations": ["training cost", "edge latency"],
        },
        {
            "family": "transformer",
            "examples": ["Transformer", "Informer", "Autoformer"],
            "strengths": ["long-horizon", "global attention"],
            "limitations": ["memory", "hyperparameter tuning"],
        },
        {
            "family": "mv_transformer",
            "examples": ["MV-Transformer", "cross-metric attention"],
            "strengths": ["CPU/memory/network joint forecast", "MAPE integration"],
            "limitations": ["high compute", "noisy high-dimensional metrics"],
        },
    ]


def select_model_stub(workload_regime: str) -> str:
    """Model Control System min-heap stub: pick forecaster by regime."""
    regime = workload_regime.lower()
    if "long" in regime or "seasonal" in regime:
        return "informer"
    if "bursty" in regime or "multivariate" in regime:
        return "mv_transformer"
    if "edge" in regime or "lightweight" in regime:
        return "lstm"
    return "arima"
