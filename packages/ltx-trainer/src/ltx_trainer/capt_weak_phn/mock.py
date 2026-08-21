"""CAPT weak-phoneme GOP smoke (arXiv:2605.23593)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.capt_weak_phn.config import CaptWeakPhnConfig
from ltx_trainer.capt_weak_phn.gop import gop_feature_vector, gop_from_log_posteriors


def evaluation_smoke(cfg: CaptWeakPhnConfig | None = None) -> dict[str, Any]:
    c = cfg or CaptWeakPhnConfig()
    rng = np.random.default_rng(0)
    k = c.inventory_k
    log_p = np.log(rng.dirichlet(np.ones(k), size=12) + 1e-8)
    target = 2
    gop = gop_from_log_posteriors(log_p, target)
    feat = gop_feature_vector(log_p, target)
    return {
        "paper": c.paper_arxiv,
        "gop_score": round(gop, 4),
        "gop_feature_dim": int(feat.size),
        "feature_dim": int(feat.size),
    }
