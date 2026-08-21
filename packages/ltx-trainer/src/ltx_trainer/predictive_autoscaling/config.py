"""Predictive autoscaling survey configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.predictive_autoscaling.constants import (
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)


@dataclass
class PredictiveAutoscalingConfig:
    """Default parameters for MAPE / CRD demo (Appendix A)."""

    pod_capacity_rpm: float = 1000.0  # requests per minute per pod
    rrs_factor: float = 0.6  # Resource Removal Strategy γ
    forecast_horizon_min: int = 5
    model_type: str = "informer"  # informer | mv_transformer | lstm
    min_replicas: int = 2
    max_replicas: int = 50
    adi_threshold: float = 0.15  # τ_ADI drift correction trigger
    frsc_epsilon: float = 0.2  # straggler tolerance vs median τ
    correction_gain_alpha: float = 0.5
    correction_cap_kappa: float = 0.3


__all__ = ["PredictiveAutoscalingConfig", "PAPER_ARXIV", "PAPER_TITLE", "PAPER_URL"]
