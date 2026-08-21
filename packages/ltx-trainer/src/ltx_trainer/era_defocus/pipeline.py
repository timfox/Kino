"""Synthetic defocus deblur demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.era_defocus.benchmarks import summary_anchors
from ltx_trainer.era_defocus.config import EraDefocusConfig
from ltx_trainer.era_defocus.losses import era_training_loss, psnr
from ltx_trainer.era_defocus.model import era_forward_stub


def _synthetic_defocus_pair(length: int = 32, seed: int = 0) -> tuple[list[float], list[float], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    gt = np.sin(np.linspace(0, 4 * np.pi, length)).tolist()
    kernel = [0.25, 0.5, 0.25]
    obs = []
    for i in range(length):
        acc = sum(kernel[j] * gt[(i + j) % length] for j in range(len(kernel)))
        obs.append(float(acc + rng.normal(0, 0.02)))
    return gt, obs, kernel, gt


def run_demo(cfg: EraDefocusConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EraDefocusConfig()
    gt, obs, kernel, _ = _synthetic_defocus_pair()
    out = era_forward_stub(obs, cfg)
    pred = out["pred"]
    loss = era_training_loss(pred, gt, obs, kernel, omega=cfg.omega_recon)
    pred_psnr = psnr(pred, gt)
    summary = summary_anchors()
    return {
        "config": {
            "unrolling_depth": cfg.unrolling_depth,
            "omega_recon": cfg.omega_recon,
        },
        "loss": round(loss, 6),
        "psnr_toy": round(pred_psnr, 3),
        "error_e": round(float(out["error_e"]), 6),
        "beats_irnext": summary["gain_vs_best_baseline_db"] > 0,
        "paper_psnr_anchor": summary["DPDD_PSNR"],
        "summary": summary,
    }
