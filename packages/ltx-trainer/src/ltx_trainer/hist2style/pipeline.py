"""Hist2Style framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.hist2style.config import Hist2StyleConfig
from ltx_trainer.hist2style.histogram import apply_sliders, marginal_histogram
from ltx_trainer.hist2style.losses import marginal_wasserstein_loss
from ltx_trainer.hist2style.model import Hist2Style


def framework_card(cfg: Hist2StyleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Hist2StyleConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "url": cfg.project_url,
        "params_m": cfg.params_m,
        "teacher": cfg.teacher,
        "architecture": {
            "style": "marginal Y/U/V histogram embedding",
            "transform": f"bilateral grid ({cfg.grid_guidance}×{cfg.grid_height}×{cfg.grid_width})",
            "constraints": "locally affine, edge-aware (no hallucination)",
        },
        "training_pairs_m": cfg.training_pairs_m,
    }


def table1_user_study() -> list[dict[str, Any]]:
    """Table 1 — H2S win/tie/lose % vs baselines."""
    return [
        {"method": "SA-LUT", "h2s_win": 82.57, "tie": 3.56, "lose": 13.86},
        {"method": "WCT2", "h2s_win": 72.58, "tie": 5.85, "lose": 21.57},
        {"method": "Xia et al.", "h2s_win": 73.24, "tie": 4.10, "lose": 22.66},
        {"method": "IDT", "h2s_win": 73.75, "tie": 3.01, "lose": 23.25},
        {"method": "D-LUT", "h2s_win": 70.59, "tie": 3.04, "lose": 26.37},
        {"method": "PhotoWCT2", "h2s_win": 61.62, "tie": 6.26, "lose": 32.12},
    ]


def table2_runtime_s() -> list[dict[str, Any]]:
    """Table 2 — runtime (seconds) at resolutions."""
    res = [256, 512, 1024, 2048, 4096]
    h2s = [0.001, 0.003, 0.009, 0.04, 0.1]
    xia = [0.003, 0.003, 0.004, 0.008, 0.03]
    return [
        {"resolution": r, "Hist2Style": t, "Xia_et_al": x}
        for r, t, x in zip(res, h2s, xia)
    ]


def table3_metrics() -> list[dict[str, Any]]:
    """Table 3 — quantitative metrics (N=7000)."""
    return [
        {"method": "Hist2Style", "user_score": 50.00, "sqa": 50.00, "cycle_mse": 401.92, "fid": 71.44},
        {"method": "PhotoWCT2", "user_score": 35.25, "sqa": 36.70, "cycle_mse": 1012.90, "fid": 112.07},
        {"method": "Xia et al.", "user_score": 24.71, "sqa": 24.12, "cycle_mse": 646.40, "fid": 50.46},
        {"method": "D-LUT", "user_score": 27.89, "sqa": 25.30, "cycle_mse": 430.35, "fid": 87.65},
        {"method": "IDT", "user_score": 24.75, "sqa": 25.70, "cycle_mse": 215.57, "fid": 92.32},
    ]


def forward_smoke(cfg: Hist2StyleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Hist2StyleConfig()
    model = Hist2Style(cfg)
    content = torch.rand(1, 3, 128, 128)
    style = torch.rand(1, 3, 64, 64)
    hist = marginal_histogram(style, cfg.hist_bins)
    out = model(content, hist)
    gt = torch.rand_like(content)
    w_loss = float(marginal_wasserstein_loss(out.detach(), gt))
    edited = apply_sliders(hist, exposure=1.1, contrast=0.9)
    return {
        "content_shape": list(content.shape),
        "output_shape": list(out.shape),
        "wasserstein_loss": w_loss,
        "hist_edited_shape": list(edited.shape),
        "params_m": cfg.params_m,
    }


def evaluation_demo(cfg: Hist2StyleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Hist2StyleConfig()
    return {
        "framework": framework_card(cfg),
        "table1_user_study": table1_user_study(),
        "table2_runtime": table2_runtime_s(),
        "table3_metrics": table3_metrics(),
        "forward": forward_smoke(cfg),
    }
