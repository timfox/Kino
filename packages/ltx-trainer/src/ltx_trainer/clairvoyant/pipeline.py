"""End-to-end CPU demo: features → predict → SJF dispatch."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clairvoyant.config import ClairvoyantConfig
from ltx_trainer.clairvoyant.features import extract_features
from ltx_trainer.clairvoyant.predictor import predict_record
from ltx_trainer.clairvoyant.queueing import discrete_event_simulation, fcfs_mean_waiting_time
from ltx_trainer.clairvoyant.scheduler import sjf_dispatch_order, validate_short_before_long


# Representative Dolly-style prompts (Sec. 5.4 dispatch test)
_SHORT_PROMPTS = (
    "What is the capital of France?",
    "Define photosynthesis in one sentence.",
    "Who wrote Pride and Prejudice?",
    "List three primary colors.",
)
_LONG_PROMPTS = (
    "Write a detailed creative story about a lighthouse keeper discovering a hidden underwater city.",
    "Explain step by step how to implement a binary search tree in Python with full class definitions.",
    "Generate a comprehensive markdown tutorial on distributed systems consensus algorithms.",
    "Describe in depth the historical causes and consequences of the Industrial Revolution.",
)


def evaluation_demo_run(*, variant: str = "sharegpt") -> dict[str, Any]:
    cfg = ClairvoyantConfig(model_variant=variant)
    short_ids = [f"short_{i}" for i in range(len(_SHORT_PROMPTS))]
    long_ids = [f"long_{i}" for i in range(len(_LONG_PROMPTS))]
    arrivals: list[tuple[str, str, float]] = []
    t = 0.0
    for sid, p in zip(short_ids, _SHORT_PROMPTS, strict=True):
        arrivals.append((sid, p, t))
        t += 0.01
    for lid, p in zip(long_ids, _LONG_PROMPTS, strict=True):
        arrivals.append((lid, p, t))
        t += 0.01

    order = sjf_dispatch_order(arrivals, starvation_tau_s=cfg.starvation_tau_s, variant=variant)
    validation = validate_short_before_long(short_ids, long_ids, order)

    sample_feat = extract_features(_LONG_PROMPTS[1])
    sim = discrete_event_simulation(n_requests=500, policy="sjf", seed=0, starvation_tau_s=cfg.starvation_tau_s)
    fcfs_w = fcfs_mean_waiting_time(0.74, 6.2, 1.03)

    predictions = {
        "short": [predict_record(p, variant=variant) for p in _SHORT_PROMPTS],
        "long": [predict_record(p, variant=variant) for p in _LONG_PROMPTS],
    }

    return {
        "config": {
            "variant": variant,
            "starvation_tau_s": cfg.starvation_tau_s,
            "mu_short_s": cfg.mu_short_s,
        },
        "feature_sample": {k: sample_feat[k] for k in sample_feat if k != "vector"},
        "predictions": predictions,
        "dispatch_validation": validation,
        "queue_sim_snippet": sim,
        "fcfs_pk_wait_s_rho074": round(fcfs_w, 2),
    }
