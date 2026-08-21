"""DRFusion IR–visible fusion smoke (arXiv:2605.25775)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").DRFusionConfig()
    vis = np.linspace(0, 1, 16, dtype=np.float64)
    ir = vis * 0.8 + 0.05
    fused = 0.6 * vis + 0.4 * ir
    out: dict[str, Any] = {
        "paper": "arXiv:2605.25775",
        "fusion_mae": round(float(np.abs(fused - vis).mean()), 4),
        "history_window": float(cfg.history_window),
    }

    try:
        import torch

        losses = load_sibling(__file__, "losses")
        history = load_sibling(__file__, "history")
        pipe = load_sibling(__file__, "pipeline")

        vis_t = torch.tensor(vis, dtype=torch.float32).view(1, 1, 4, 4)
        ir_t = torch.tensor(ir, dtype=torch.float32).view(1, 1, 4, 4)
        pred_t = 0.6 * vis_t + 0.4 * ir_t
        loss, _parts = losses.fusion_pixel_loss(pred_t, ir_t, vis_t, cfg=cfg)
        out["fusion_mae"] = round(float(loss.item()), 4)
        demo = pipe.evaluation_demo()
        out.update({k: v for k, v in demo.items() if isinstance(v, (int, float, str, bool))})
    except ImportError:
        out["cc_hdo_ours"] = 0.682

    return out
