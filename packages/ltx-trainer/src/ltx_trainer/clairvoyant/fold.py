"""Fold scheduling metadata for prompt sidecars in AV training."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clairvoyant.predictor import predict_record


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption")
        or meta.get("prompt")
        or meta.get("text")
        or data.get("caption")
        or ""
    )
    if caption.strip():
        pred = predict_record(caption)
    else:
        pred = {"p_long": 0.5, "predicted_class": "medium", "prompt_token_len": 0}
    p_long = float(pred.get("p_long", 0.5))
    holb_risk = "high" if p_long >= 0.6 else ("low" if p_long <= 0.25 else "medium")
    out["clairvoyant"] = {
        "arxiv_id": "2606.07248",
        "p_long": p_long,
        "predicted_length_class": pred.get("predicted_class", "medium"),
        "prompt_token_len_proxy": pred.get("prompt_token_len", 0),
        "holb_risk_if_serial_fcfs": holb_risk,
        "scheduler_hint": "sjf_priority_ascending_p_long",
    }
    return out
