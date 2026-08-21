"""ChWDTA framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.chwdta.charm import charm_smoke
from ltx_trainer.chwdta.chwdta import ChWDTB
from ltx_trainer.chwdta.chwp import chwp_summary
from ltx_trainer.chwdta.config import ChwdtaConfig
from ltx_trainer.chwdta.covariance import roff_block_ratio
import numpy as np


def framework_card(cfg: ChwdtaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChwdtaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "repo": cfg.repo_url,
        "modules": {
            "ChWDTB": "WTc → windowed spatial MSA → IWTc",
            "ChWP": "two-level channel wavelet packet → 4 subbands",
            "ChARM": "slice-sequential Gaussian entropy + LRP",
        },
        "variants": {
            "8-slice": {"slices": cfg.entropy_slices_8, "default": True},
            "4-slice": {"slices": cfg.entropy_slices_4, "lower_latency": True},
        },
        "headline_bd_rate_8": {
            "Kodak": cfg.bd_rate_kodak_8,
            "CLIC": cfg.bd_rate_clic_8,
            "Tecnick": cfg.bd_rate_tecnick_8,
        },
    }


def table1_complexity(cfg: ChwdtaConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or ChwdtaConfig()
    return [
        {"model": "DCAE (CVPR'25)", "bd_kodak": cfg.dcae_bd_kodak, "latency_ms": 184, "params_m": 119.4},
        {
            "model": "Ours (4-slice)",
            "bd_kodak": cfg.bd_rate_kodak_4,
            "latency_ms": cfg.latency_4_ms,
            "params_m": cfg.params_4_m,
        },
        {
            "model": "Ours (8-slice)",
            "bd_kodak": cfg.bd_rate_kodak_8,
            "latency_ms": cfg.latency_8_ms,
            "params_m": cfg.params_8_m,
        },
    ]


def table2_ablation() -> list[dict[str, Any]]:
    """Paper Table II — module ablation Kodak BD-rate."""
    return [
        {"method": "Ours (8-slice)", "bd_rate": -17.82},
        {"method": "w/o TB Lift", "bd_rate": -16.76},
        {"method": "w/o TB Lift, ChWP", "bd_rate": -15.50},
    ]


def forward_smoke() -> dict[str, Any]:
    block = ChWDTB(64, num_heads=8)
    x = torch.randn(1, 64, 32, 32)
    y = block(x)
    return {"in_shape": list(x.shape), "out_shape": list(y.shape)}


def evaluation_demo(cfg: ChwdtaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChwdtaConfig()
    rng = np.random.default_rng(60111)
    cov = rng.normal(size=(48, 48)).astype(np.float64)
    cov = cov @ cov.T
    w = np.eye(48)  # identity as placeholder WT
    cov_w = w @ cov @ w.T
    return {
        "framework": framework_card(cfg),
        "table1": table1_complexity(cfg),
        "table2_ablation": table2_ablation(),
        "forward": forward_smoke(),
        "chwp": chwp_summary(cfg),
        "charm": charm_smoke(8),
        "roff_native": roff_block_ratio(cov),
        "roff_wt": roff_block_ratio(cov_w),
    }
