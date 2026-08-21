"""Pathos-MM Russell projection + correlation smoke (arXiv:2605.22732)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pathos_mm.config import PathosMmConfig
from ltx_trainer.pathos_mm.metrics import spearman_rho
from ltx_trainer.pathos_mm.russell import E2V_CLASSES, russell_arousal_valence


def evaluation_smoke(cfg: PathosMmConfig | None = None) -> dict[str, Any]:
    c = cfg or PathosMmConfig()
    probs = {cls: 1.0 / len(E2V_CLASSES) for cls in E2V_CLASSES}
    probs["happy"] = 0.4
    probs["neutral"] = 0.1
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    arousal, valence = russell_arousal_valence(probs)
    trust = np.array([0.5, 0.8, 1.0, 0.3, 0.9])
    valence_track = np.array([valence, 0.2, 0.6, -0.1, 0.4])
    rho = spearman_rho(valence_track, trust)
    return {
        "paper": c.paper_arxiv,
        "paper_rho_gemini_valence_trust": c.rho_gemini_valence_trust,
        "toy_russell_arousal": round(arousal, 3),
        "toy_russell_valence": round(valence, 3),
        "toy_spearman_rho": round(float(rho), 3) if not np.isnan(rho) else None,
    }
