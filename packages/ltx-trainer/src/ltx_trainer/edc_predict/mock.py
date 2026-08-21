"""EDC prediction log-domain loss smoke (arXiv:2605.20968)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.edc_predict.config import EdcPredictConfig
from ltx_trainer.edc_predict.loss import composite_loss
from ltx_trainer.edc_predict.rss import random_sign_sticky_sequence


def evaluation_smoke(cfg: EdcPredictConfig | None = None) -> dict[str, Any]:
    c = cfg or EdcPredictConfig()
    t = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    y = np.exp(-3.0 * t)
    y_hat = y * 0.95 + 0.01
    loss = composite_loss(y_hat, y, alpha=c.loss_alpha, stride_k=c.slope_stride_k)
    signs = random_sign_sticky_sequence(32, stickiness_p=c.rss_stickiness_p, rng=np.random.default_rng(0))
    reduction = 100.0 * (1.0 - c.convnet_params_m / c.lstm_params_m)
    return {
        "paper": c.paper_arxiv,
        "param_reduction_pct": reduction,
        "bands": c.n_third_octave_bands,
        "L_total": round(loss["L_total"], 4),
        "mse_db": round(loss["mse_db"], 4),
        "rss_flip_rate": round(float(np.mean(signs[1:] != signs[:-1])), 3),
    }
