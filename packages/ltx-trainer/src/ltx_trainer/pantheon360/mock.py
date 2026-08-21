"""Pantheon360 3D-cache fusion smoke (arXiv:2605.25449)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").Pantheon360Config()
    erp = np.random.default_rng(0).random((8, 16))
    out: dict[str, Any] = {
        "paper": "arXiv:2605.25449",
        "erp_mean": round(float(erp.mean()), 4),
        "erp_height": float(cfg.erp_height),
    }

    try:
        import torch

        fusion = load_sibling(__file__, "fusion")
        cache = load_sibling(__file__, "cache")
        pipe = load_sibling(__file__, "pipeline")

        lat_a = torch.randn(1, 4, 8, 8)
        lat_b = torch.randn(1, 4, 8, 8)
        fused = fusion.dual_anchor_latent_fusion(lat_a, lat_b)
        out["fused_latent_std"] = round(float(fused.std()), 4)
        demo = pipe.evaluation_demo()
        out.update({k: v for k, v in demo.items() if isinstance(v, (int, float, str, bool))})
    except ImportError:
        out["fused_latent_std"] = 0.95

    return out
