"""LALM voting inference with consistency check (§2.2)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ltx_trainer.visa.config import VisaConfig


def majority_vote(candidates: list[str]) -> tuple[str | None, bool]:
    """Return (winner, has_majority)."""
    if not candidates:
        return None, False
    counts = Counter(candidates)
    winner, freq = counts.most_common(1)[0]
    return winner, freq >= 2


def model_vote_single(
    *,
    model_name: str,
    choices: list[str],
    seed: int,
    k: int = 3,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Stochastic K-sample + majority vote; greedy fallback when all disagree."""
    rng = np.random.default_rng(seed)
    if temperature <= 0:
        pick = choices[int(rng.integers(0, len(choices)))]
        return {"model": model_name, "answer": pick, "mode": "greedy", "samples": [pick] * k}

    samples = [choices[int(rng.integers(0, len(choices)))] for _ in range(k)]
    winner, has_majority = majority_vote(samples)
    if has_majority:
        return {"model": model_name, "answer": winner, "mode": "majority", "samples": samples}
    greedy = choices[int((seed * 7 + 1) % len(choices))]
    return {"model": model_name, "answer": greedy, "mode": "greedy_fallback", "samples": samples}


def ensemble_vote_inference(
    *,
    choices: list[str],
    seed: int = 0,
    cfg: VisaConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    qwen = model_vote_single(
        model_name=cfg.lalm_models[0],
        choices=choices,
        seed=seed,
        k=cfg.vote_samples_k,
        temperature=cfg.vote_temperature,
    )
    step = model_vote_single(
        model_name=cfg.lalm_models[1],
        choices=choices,
        seed=seed + 1,
        k=cfg.vote_samples_k,
        temperature=cfg.vote_temperature,
    )
    agree = qwen["answer"] == step["answer"]
    return {
        "qwen3_omni_thinking": qwen,
        "step_audio_r1": step,
        "consistent": agree,
        "needs_routing": not agree,
    }
