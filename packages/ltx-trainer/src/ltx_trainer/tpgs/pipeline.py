"""Training demo and ablation checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.tpgs.benchmarks import TABLE2_RICOH360, TABLE4_ABLATION, ours_beats_odgs
from ltx_trainer.tpgs.config import TpgsConfig
from ltx_trainer.tpgs.metrics import lpips_proxy, psnr, ssim_proxy
from ltx_trainer.tpgs.tpgs_net import TpgsStub
from ltx_trainer.tpgs.transition_plane import view_rotations


def evaluation_demo_run(cfg: TpgsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TpgsConfig(cube_face_res=32, intra_steps=2, inter_steps=1)
    model = TpgsStub(cfg)
    with torch.no_grad():
        out = model()
    n_views = len(view_rotations(include_transition=cfg.use_transition_plane))
    return {
        "erp_shape": list(out["erp_rgb"].shape),
        "num_views": n_views,
        "loss": float(out["loss"].item()),
        "beats_odgs_psnr_ricoh": ours_beats_odgs("Ricoh360", "10min", "PSNR"),
        "center_ours_psnr": TABLE2_RICOH360["center_10min"]["Ours"]["PSNR"],
    }


def train_step(cfg: TpgsConfig | None = None) -> dict[str, float]:
    cfg = cfg or TpgsConfig(cube_face_res=16, use_transition_plane=True, use_intra_inter=True)
    model = TpgsStub(cfg)
    out = model()
    out["loss"].backward()
    return {
        "loss": float(out["loss"].detach()),
        "loss_intra": float(out["loss_intra"]),
        "loss_inter": float(out["loss_inter"]),
    }


def nvs_metrics_demo() -> dict[str, float]:
    pred = torch.rand(1, 3, 64, 128)
    gt = pred + 0.05 * torch.randn_like(pred)
    return {
        "psnr": float(psnr(pred.clamp(0, 1), gt.clamp(0, 1)).item()),
        "ssim": float(ssim_proxy(pred, gt).item()),
        "lpips": float(lpips_proxy(pred, gt).item()),
    }


def ablation_table_check() -> dict[str, bool]:
    c = TABLE4_ABLATION["center"]
    return {
        "tp_improves_over_3dgs_p": c["TP"]["PSNR"] > c["3DGS(P)"]["PSNR"],
        "op_improves_over_tp": c["TP+OP"]["PSNR"] > c["TP"]["PSNR"],
        "full_best_psnr": c["TP+OP+CP"]["PSNR"] >= c["TP+OP"]["PSNR"],
        "full_100min_best": c["full_100min"]["PSNR"] > c["TP+OP+CP"]["PSNR"],
    }
