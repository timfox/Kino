"""Plug-in EDL loss smoke (arXiv:2605.22746)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.plug_edl.config import PlugEdlConfig
from ltx_trainer.plug_edl.evidential import dirichlet_params, project_dirichlet, softmax_from_logits
from ltx_trainer.plug_edl.losses import plug_in_cross_entropy


def evaluation_smoke(cfg: PlugEdlConfig | None = None) -> dict[str, Any]:
    c = cfg or PlugEdlConfig()
    logits = np.array([2.0, 0.5, -1.0, -0.5], dtype=np.float64)
    p_hat = softmax_from_logits(logits)
    alpha = dirichlet_params(softmax_from_logits(logits) * 10.0, prior=c.evidence_prior)
    _ = project_dirichlet(alpha)
    ce = plug_in_cross_entropy(p_hat, y=0)
    return {
        "paper": c.paper_arxiv,
        "plug_ce": round(ce, 4),
        "paper_softmax_base_acc_pct": c.softmax_base_acc_pct,
        "predicted_class": int(p_hat.argmax()),
    }
