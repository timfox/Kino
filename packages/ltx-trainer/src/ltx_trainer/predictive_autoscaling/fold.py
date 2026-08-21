"""AV-fold sidecar: predictive autoscaling metadata on captions."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.config import PredictiveAutoscalingConfig
from ltx_trainer.predictive_autoscaling.constants import PAPER_ARXIV
from ltx_trainer.predictive_autoscaling.drift import autoscaling_drift_index
from ltx_trainer.predictive_autoscaling.models import select_model_stub
from ltx_trainer.predictive_autoscaling.pipeline import plan_replicas, toy_forecast


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach workload tier + drift hint for merged_native captions."""
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption")
        or meta.get("prompt")
        or meta.get("text")
        or data.get("caption")
        or ""
    )
    words = len(caption.split())
    last_rpm = 800.0 + words * 12.0
    regime = "bursty multivariate" if words > 40 else "edge lightweight"
    cfg = PredictiveAutoscalingConfig()
    forecast = toy_forecast(last_rpm)
    plan = plan_replicas(forecast, cfg, current_replicas=4)
    obs = [0.6 + 0.05 * i for i in range(4)]
    pred = [0.58 + 0.04 * i for i in range(4)]
    adi = autoscaling_drift_index(obs, pred)
    out["predictive_autoscaling"] = {
        "arxiv_id": PAPER_ARXIV,
        "forecast_model": select_model_stub(regime),
        "forecast_rpm": round(forecast, 2),
        "desired_replicas": plan["desired_after_rrs"],
        "adi": round(adi, 4),
        "scaler": "PredictiveAutoscaler CRD" if adi < cfg.adi_threshold else "hybrid+ADI correction",
        "regime": regime,
    }
    return out
