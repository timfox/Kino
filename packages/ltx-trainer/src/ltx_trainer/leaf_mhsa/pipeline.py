"""Framework card, Table 2, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.leaf_mhsa.config import LeafMHSAConfig
from ltx_trainer.leaf_mhsa.metrics import mae_per_band, nrmse_percent, r2_score
from ltx_trainer.leaf_mhsa.model import TraitToSpectraMHSA
from ltx_trainer.leaf_mhsa.prospect import prospect_pro_forward
from ltx_trainer.leaf_mhsa.traits import TRAIT_NAMES


def framework_card(cfg: LeafMHSAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LeafMHSAConfig()
    return {
        "doi": cfg.doi,
        "title": cfg.title,
        "venue": cfg.venue,
        "task": "16 grapevine leaf traits → hyperspectral reflectance (400–2500 nm)",
        "n_traits": cfg.n_traits,
        "trait_names": list(TRAIT_NAMES),
        "n_bands": cfg.n_bands,
        "architecture": {
            "attention_heads": cfg.n_heads,
            "embed_dim": cfg.embed_dim,
            "conv": f"{cfg.conv_filters[0]}/{cfg.conv_filters[1]} filters, k={cfg.conv_kernel}",
            "baseline": "PROSPECT-PRO forward (generalized RTM)",
        },
        "dataset": {
            "n_leaves": cfg.n_leaves,
            "n_unique": cfg.n_unique_samples,
            "varieties": list(cfg.varieties),
            "years": list(cfg.collection_years),
        },
        "cv_folds": cfg.cv_folds,
    }


def table2_cross_validation() -> list[dict[str, Any]]:
    """Table 2 — stratified 5-fold CV R² and NRMSE (%)."""
    rows = [
        {"fold": 1, "r2": 0.81, "nrmse_pct": 1.61},
        {"fold": 2, "r2": 0.84, "nrmse_pct": 1.72},
        {"fold": 3, "r2": 0.79, "nrmse_pct": 2.02},
        {"fold": 4, "r2": 0.85, "nrmse_pct": 1.12},
        {"fold": 5, "r2": 0.89, "nrmse_pct": 1.12},
    ]
    rows.append(
        {
            "fold": "average",
            "r2": round(sum(r["r2"] for r in rows) / len(rows), 2),
            "nrmse_pct": round(sum(r["nrmse_pct"] for r in rows) / len(rows), 2),
        }
    )
    return rows


def table2_summary() -> dict[str, float]:
    avg = table2_cross_validation()[-1]
    return {"r2": float(avg["r2"]), "nrmse_pct": float(avg["nrmse_pct"])}


def forward_smoke(cfg: LeafMHSAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LeafMHSAConfig()
    n = cfg.demo_n_bands
    model = TraitToSpectraMHSA(cfg, n_bands=n)
    traits = torch.rand(4, cfg.n_traits)
    measured = torch.rand(4, n) * 0.4 + 0.1
    pred = model(traits)
    pp = prospect_pro_forward(traits, n)
    return {
        "traits_shape": list(traits.shape),
        "pred_shape": list(pred.shape),
        "r2": float(r2_score(measured.flatten(), pred.flatten()).detach()),
        "nrmse_pct": float(nrmse_percent(measured.flatten(), pred.flatten()).detach()),
        "mae_mhsa_mean": float(mae_per_band(measured, pred).mean().detach()),
        "mae_prospect_mean": float(mae_per_band(measured, pp).mean().detach()),
        "demo_n_bands": n,
    }


def evaluation_demo(cfg: LeafMHSAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LeafMHSAConfig()
    return {
        "framework": framework_card(cfg),
        "table2_cv": table2_cross_validation(),
        "table2_summary": table2_summary(),
        "forward": forward_smoke(cfg),
    }
