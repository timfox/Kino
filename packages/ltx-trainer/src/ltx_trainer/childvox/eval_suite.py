"""Toy evaluation suite over ChildVox-Balanced-style tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.childvox.config import ChildVoxConfig
from ltx_trainer.childvox.datasets import balanced_distribution, dataset_registry
from ltx_trainer.childvox.encoder_proxy import layer_hiddens_from_waveform
from ltx_trainer.childvox.models import weighted_encoder_pool


def _synthetic_wave(task_id: int, *, sr: float, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + task_id)
    t = np.arange(int(sr * duration_s)) / sr
    freq = 200.0 + 30.0 * (task_id % 7)
    return (0.4 + 0.1 * rng.random()) * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.01, t.size)


def run_balanced_eval(
    *,
    n_per_task: int = 8,
    seed: int = 42,
    cfg: ChildVoxConfig | None = None,
) -> dict[str, Any]:
    """Classify synthetic clips with weighted encoder pool; report macro accuracy."""
    cfg = cfg or ChildVoxConfig()
    sr = cfg.sample_rate_hz
    tasks = balanced_distribution()[:6]
    correct = 0
    total = 0
    per_dataset: dict[str, float] = {}
    for ti, row in enumerate(tasks):
        ds_name = row["dataset"]
        n_classes = 5
        hits = 0
        for j in range(n_per_task):
            wave = _synthetic_wave(ti * 10 + j, sr=sr, duration_s=0.5, seed=seed)
            h = layer_hiddens_from_waveform(wave, sample_rate=sr)
            gold = (ti + j) % n_classes
            pred = weighted_encoder_pool(h, num_classes=n_classes, seed=seed + j)["pred"]
            hits += int(pred == gold)
            total += 1
            correct += int(pred == gold)
        per_dataset[ds_name] = 100.0 * hits / max(n_per_task, 1)
    return {
        "macro_accuracy_pct": 100.0 * correct / max(total, 1),
        "n_samples": total,
        "n_datasets": len(tasks),
        "per_dataset_accuracy": per_dataset,
        "registry_count": len(dataset_registry()),
    }


def eval_suite_smoke(cfg: ChildVoxConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or ChildVoxConfig()
    out = run_balanced_eval(n_per_task=4, seed=seed, cfg=cfg)
    return {
        "macro_accuracy_pct": out["macro_accuracy_pct"],
        "n_samples": out["n_samples"],
        "registry_count": out["registry_count"],
        "computed_not_config": True,
    }
