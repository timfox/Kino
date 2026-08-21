"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.rigpapr.benchmarks import benchmarks_bundle, table1_ours
from ltx_trainer.rigpapr.config import RigPAPRConfig
from ltx_trainer.rigpapr.integration import integration_bundle
from ltx_trainer.rigpapr.lbs import idw_transfer, lbs_demo, lbs_deform, rodrigues, se3_from_rotation, softmax_weights
from ltx_trainer.rigpapr.optimization import optimization_bundle, phase1_step_stub, phase2_step_stub
from ltx_trainer.rigpapr.papr import papr_render_demo, render_patch


def framework_card(cfg: RigPAPRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RigPAPRConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "representation": "PAPR proximity-attention points (no per-primitive covariance)",
            "rigging": "OffsetOPT mesh proxy → Puppeteer auto-rig → IDW weight transfer",
            "deformation": "Direct LBS on positions; ui, τi rigid",
            "driving": "Fixed-viewpoint monocular video (captured or I2V)",
            "optimization": "Two-phase: depth-supervised rotations → track+ARAP+weights",
        },
        "config": cfg.__dict__,
        "optimization": optimization_bundle(),
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    cfg = RigPAPRConfig(num_points=256, num_bones=12)
    rng = np.random.default_rng(seed)
    canon = rng.standard_normal((cfg.num_points, 3)) * 0.4
    mesh_v = rng.standard_normal((64, 3)) * 0.4
    mesh_w = softmax_weights(rng.standard_normal((64, cfg.num_bones)))
    weights = idw_transfer(mesh_v, mesh_w, canon, k=cfg.idw_k)
    angles = rng.standard_normal((cfg.num_bones, 3)) * 0.15
    T = np.stack([se3_from_rotation(rodrigues(angles[j])) for j in range(cfg.num_bones)])
    deformed = lbs_deform(canon, weights, T)
    feat = rng.standard_normal((cfg.num_points, cfg.feature_dim))
    canon_rgb = render_patch(canon, feat, height=24, width=24, top_k=cfg.top_k, seed=seed)
    def_rgb = render_patch(deformed, feat, height=24, width=24, top_k=cfg.top_k, seed=seed + 1)
    p1 = phase1_step_stub(canon, weights, angles, seed=seed)
    logits = rng.standard_normal((cfg.num_points, cfg.num_bones))
    corr = rng.standard_normal((cfg.num_bones, 3)) * 0.02
    p2 = phase2_step_stub(canon, logits, angles, corr, seed=seed)
    ours = table1_ours()
    return {
        "lbs": lbs_demo(seed=seed, num_points=cfg.num_points, num_bones=cfg.num_bones),
        "papr": papr_render_demo(seed=seed, num_points=cfg.num_points),
        "phase1_losses": p1,
        "phase2_losses": p2,
        "render_delta": float(np.abs(canon_rgb - def_rgb).mean()),
        "ref_novel_psnr": ours["synth_novel_psnr"],
        "novel_gain_db": 23.63 - 20.51,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    ours = table1_ours()
    return {
        "paper": "rigpapr",
        "arxiv": "2606.06685",
        "ref_train_psnr": ours["synth_train_psnr"],
        "ref_novel_psnr": ours["synth_novel_psnr"],
        "novel_psnr_margin_db": demo["novel_gain_db"],
        "phase1_total": demo["phase1_losses"]["total"],
        "phase2_total": demo["phase2_losses"]["total"],
        "render_delta": demo["render_delta"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
