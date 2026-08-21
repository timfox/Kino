"""Stratified evaluation on the 1,500-item VOICEGIRAFFE pool."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.voicegiraffe.config import InferenceMode, VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.dataset import QAItem
from ltx_trainer.voicegiraffe.eval import run_eval
from ltx_trainer.voicegiraffe.hub_audio import load_recording_waves
from ltx_trainer.voicegiraffe.recording_registry import build_recording_catalog, load_full_qa_pool


def stratified_sample(
    items: list[QAItem],
    *,
    n_per_tier: int = 32,
    seed: int = 42,
) -> list[QAItem]:
    """Draw a balanced single-hop / multi-hop subset."""
    rng = np.random.default_rng(seed)
    by_tier: dict[str, list[QAItem]] = {}
    for item in items:
        by_tier.setdefault(item.tier, []).append(item)
    out: list[QAItem] = []
    for tier in sorted(by_tier):
        pool = by_tier[tier]
        idx = rng.choice(len(pool), size=min(n_per_tier, len(pool)), replace=False)
        out.extend(pool[int(i)] for i in idx)
    return out


def run_sampled_pool_eval(
    *,
    n_per_tier: int = 32,
    mode: InferenceMode = InferenceMode.CASCADE,
    cfg: VoiceGiraffeConfig | None = None,
    seed: int = 42,
    use_hub_audio: bool = True,
) -> dict[str, Any]:
    """Evaluate a stratified slice of the full QA pool."""
    cfg = cfg or VoiceGiraffeConfig()
    full = load_full_qa_pool(cfg, seed=seed)
    sample = stratified_sample(full, n_per_tier=n_per_tier, seed=seed)
    catalog = build_recording_catalog(cfg, seed=seed)
    waves = load_recording_waves(catalog, seed=seed) if use_hub_audio else {}

    out = run_eval(sample, mode=mode, seed=seed, waveforms=waves if use_hub_audio else None)
    if waves:
        covered = sum(1 for i in sample if i.recording_id in waves)
        out["hub_audio_coverage"] = covered / max(len(sample), 1)
    out["pool_size"] = len(full)
    out["sample_size"] = len(sample)
    out["single_hop_in_sample"] = sum(1 for i in sample if i.tier == "single_hop")
    out["multi_hop_in_sample"] = sum(1 for i in sample if i.tier == "multi_hop")
    return out


def pool_eval_smoke(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    cascade = run_sampled_pool_eval(n_per_tier=8, mode=InferenceMode.CASCADE, cfg=cfg, seed=seed)
    lrm = run_sampled_pool_eval(n_per_tier=8, mode=InferenceMode.LRM, cfg=cfg, seed=seed + 1)
    return {
        "pool_size": cascade["pool_size"],
        "sample_size": cascade["sample_size"],
        "cascade_accuracy_pct": cascade["accuracy_pct"],
        "lrm_accuracy_pct": lrm["accuracy_pct"],
        "hub_audio_coverage": cascade.get("hub_audio_coverage", 0.0),
        "computed_pool_eval": cascade["pool_size"] == cfg.n_qa_total,
    }
