"""Paper stub smoke for EgoRelight."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ego_relight.config import EgoRelightConfig
from ltx_trainer.ego_relight.hdr import finlayson_ldr_to_hdr, optimize_hdr_color_correction
from ltx_trainer.ego_relight.pipeline import (
    framework_card,
    relight_frame,
    run_appearance_torch_smoke,
    run_perception_smoke,
)


def evaluation_smoke(cfg: EgoRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EgoRelightConfig()
    ldr = np.clip(np.random.default_rng(0).random((32, 64, 3)).astype(np.float32) * 0.8 + 0.1, 0, 1)
    A, gamma, loss = optimize_hdr_color_correction(ldr, target_mean=np.array([0.45, 0.45, 0.45], dtype=np.float32), steps=20)
    hdr = finlayson_ldr_to_hdr(ldr, A, float(gamma[0]))
    relight = relight_frame(
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.2, 0.1, 1.0], dtype=np.float32),
        albedo=0.7,
        cfg=cfg,
    )
    torch_info = run_appearance_torch_smoke()
    total_ms = sum(cfg.runtime_ms.values())
    return {
        "package": "ego_relight",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "perception": run_perception_smoke(cfg),
        "hdr_optimization_loss": loss,
        "hdr_mean": float(hdr.mean()),
        "relight_composite": relight["composite"],
        "ref_psnr_subject1": cfg.ref_psnr,
        "table4_ours_subject1": framework_card(cfg)["benchmarks"]["table4_ours"][1],
        "runtime_total_ms_paper": total_ms,
        **torch_info,
    }
