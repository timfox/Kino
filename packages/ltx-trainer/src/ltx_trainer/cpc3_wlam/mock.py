"""CPC3 WLAM word-level fusion smoke (arXiv:2605.23604)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cpc3_wlam.config import Cpc3WlamConfig
from ltx_trainer.cpc3_wlam.fusion import joint_fuse, predict_word_correctness
from ltx_trainer.cpc3_wlam.word_level import masked_bce_loss


def evaluation_smoke(cfg: Cpc3WlamConfig | None = None) -> dict[str, Any]:
    c = cfg or Cpc3WlamConfig()
    rng = np.random.default_rng(0)
    decoder = rng.standard_normal((4, 8))
    local = rng.standard_normal((4, 6))
    global_utt = rng.standard_normal(4)
    severity = rng.standard_normal(2)
    joint = joint_fuse(decoder, local, global_utt, severity)
    probs = predict_word_correctness(joint, seed=0)
    valid = np.ones(4, dtype=np.float64)
    target = (probs > 0.5).astype(np.float64)
    bce = masked_bce_loss(target, probs, valid)
    return {
        "paper": c.paper_arxiv,
        "joint_repr_dim": int(joint.shape[1]),
        "joint_eval_f1": c.best_eval_f1,
        "joint_eval_corr": c.best_eval_corr,
        "masked_bce": round(bce, 4),
        "mean_word_prob": round(float(probs.mean()), 4),
    }
