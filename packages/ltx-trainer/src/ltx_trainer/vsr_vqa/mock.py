"""QoMEX VSR-VQA correlation smoke (arXiv:2605.25940)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.smoke_util import load_sibling


def _pearson_numpy(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / denom) if denom > 1e-12 else 0.0


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").VSRVQAConfig()
    rng = np.random.default_rng(0)
    mos = np.array([2.1, 2.8, 3.4, 3.9, 4.2, 4.5])
    lpips_like = 5.0 - mos + rng.standard_normal(6) * 0.03

    out: dict[str, Any] = {
        "paper": cfg.paper_arxiv,
        "pearson_stub": round(_pearson_numpy(lpips_like, mos), 4),
        "lpips_plcc_table": 0.851,
    }

    try:
        import torch
        corr = load_sibling(__file__, "correlation")
        pipe = load_sibling(__file__, "pipeline")
        pearson_correlation = corr.pearson_correlation
        spearman_correlation = corr.spearman_correlation
        rmse = corr.rmse
        fisher_z_mean = corr.fisher_z_mean
        table_within_sequence_correlation = pipe.table_within_sequence_correlation
        evaluation_demo = pipe.evaluation_demo

        mos_t = torch.tensor(mos.tolist())
        lpips_t = torch.tensor(lpips_like.tolist())
        out["pearson_stub"] = round(pearson_correlation(lpips_t, mos_t), 4)
        out["spearman_stub"] = round(spearman_correlation(lpips_t, mos_t), 4)
        out["rmse_stub"] = round(rmse(lpips_t, mos_t), 4)
        out["fisher_z_mean"] = round(fisher_z_mean([0.85, 0.88, 0.84, 0.90]), 4)
        out["lpips_plcc_table"] = table_within_sequence_correlation()["LPIPS_Alex"]["plcc"]
        out.update({k: v for k, v in evaluation_demo().items() if isinstance(v, (int, float, str, bool))})
    except ImportError:
        out["spearman_stub"] = out["pearson_stub"]
        out["rmse_stub"] = round(float(np.sqrt(((lpips_like - mos) ** 2).mean())), 4)

    return out
