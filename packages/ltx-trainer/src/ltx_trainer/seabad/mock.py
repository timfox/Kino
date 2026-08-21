"""SEABAD smoke evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.seabad.config import SeabadConfig
from ltx_trainer.seabad.dedup import is_exact_duplicate, mel_embedding_mean_std


def evaluation_smoke(cfg: SeabadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeabadConfig()
    rng = np.random.default_rng(0)
    mel = rng.standard_normal((64, 48))
    z1 = mel_embedding_mean_std(mel)
    z2 = z1.copy()
    dup = is_exact_duplicate(z1, z2)
    clip_samples = int(cfg.clip_duration_s * cfg.sample_rate_hz)
    return {
        "duplicate_detected": dup,
        "clip_samples": clip_samples,
        "embedding_dim": int(z1.size),
        "paper_species": cfg.unique_species,
        "paper_gini_pre": cfg.gini_pre_balancing,
        "paper_gini_post": cfg.gini_post_balancing,
    }
