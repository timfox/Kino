"""MC accuracy evaluation and inference-mode stubs (VOICEGIRAFFE §4)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.voicegiraffe.cascade import cascade_evaluate, sliding_window_segments, aggregate_captions
from ltx_trainer.voicegiraffe.config import InferenceMode, VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.dataset import QAItem, load_qa_items
from ltx_trainer.voicegiraffe.lrm import lrm_evaluate


def extract_mc_answer(text: str) -> str | None:
    """Regex / alignment extraction per MMAU/MMAR practice."""
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else None


def mc_accuracy(predictions: list[str], gold: list[str]) -> float:
    if not predictions or len(predictions) != len(gold):
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold) if extract_mc_answer(p) == g)
    return 100.0 * correct / len(gold)


def inference_mode_scores(cfg: VoiceGiraffeConfig | None = None) -> dict[str, float]:
    """Figure 4 style anchors for Qwen3.5-Omni-Plus."""
    cfg = cfg or VoiceGiraffeConfig()
    return {
        InferenceMode.E2E.value: cfg.best_e2e_overall_pct,
        InferenceMode.CASCADE.value: 66.2,
        InferenceMode.LRM.value: 59.53,
    }


def duration_decay_curve() -> list[dict[str, float]]:
    """Figure 5 style: performance vs duration bucket (proprietary cascade)."""
    return [
        {"duration_bucket_min": 30, "accuracy_pct": 66.0},
        {"duration_bucket_min": 45, "accuracy_pct": 63.2},
        {"duration_bucket_min": 60, "accuracy_pct": 60.5},
    ]


def run_eval(
    items: list[QAItem] | None = None,
    *,
    mode: InferenceMode = InferenceMode.CASCADE,
    seed: int = 42,
    waveforms: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate built-in or supplied QA items with cascade/LRM predictors."""
    import numpy as np

    items = items or load_qa_items()
    sr = 48000.0
    rng = np.random.default_rng(seed)
    waves: dict[str, np.ndarray] = dict(waveforms or {})
    captions: dict[str, str] = {}
    for rec_id in {i.recording_id for i in items}:
        if rec_id in waves:
            wave = waves[rec_id]
        else:
            dur = max(i.duration_min for i in items if i.recording_id == rec_id)
            wave = rng.standard_normal(int(sr * dur * 60 * 0.01))  # short proxy for speed
            waves[rec_id] = wave
        segs = sliding_window_segments(wave, sr)
        captions[rec_id] = aggregate_captions(segs)

    if mode == InferenceMode.LRM:
        out = lrm_evaluate(items, captions)
    else:
        out = cascade_evaluate(items, waves, sr=sr)

    by_tier: dict[str, list[bool]] = {}
    for item in items:
        from ltx_trainer.voicegiraffe.cascade import cascade_predict
        from ltx_trainer.voicegiraffe.lrm import lrm_predict

        cap = captions[item.recording_id]
        pred = lrm_predict(item, cap) if mode == InferenceMode.LRM else cascade_predict(item, cap)
        by_tier.setdefault(item.tier, []).append(pred == item.gold)

    out["per_tier_accuracy"] = {k: 100.0 * sum(v) / len(v) for k, v in by_tier.items()}
    out["mode"] = mode.value
    out["n_items"] = len(items)
    return out


def eval_smoke(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    rng_gold = ["A", "B", "C", "D", "A"]
    preds_e2e = ["The answer is A", "B", "Option C", "D", "A"]
    preds_weak = ["A", "A", "A", "A", "B"]
    modes = inference_mode_scores(cfg)
    cascade_run = run_eval(mode=InferenceMode.CASCADE, seed=seed)
    from ltx_trainer.voicegiraffe.pool_eval import pool_eval_smoke

    pool = pool_eval_smoke(cfg, seed=seed)
    return {
        "mc_accuracy_e2e_proxy": mc_accuracy(preds_e2e, rng_gold),
        "mc_accuracy_weak_proxy": mc_accuracy(preds_weak, rng_gold),
        "e2e_beats_cascade": modes[InferenceMode.E2E.value] > modes[InferenceMode.CASCADE.value],
        "opensource_lrm_gain": cfg.opensource_lrm_avg_pct - cfg.opensource_cascade_avg_pct,
        "only_e2e_beats_human": cfg.best_e2e_overall_pct > cfg.human_overall_pct,
        "cascade_eval_ran": cascade_run["n_items"] > 0,
        "pool_eval_sample_size": pool["sample_size"],
        "pool_eval_computed": pool["computed_pool_eval"],
        "seed": seed,
    }
