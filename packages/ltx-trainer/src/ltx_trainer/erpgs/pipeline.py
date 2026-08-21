"""Training and evaluation demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.erpgs.benchmarks import TABLE1_NVS, TABLE2_ABLATION, ours_beats_baseline
from ltx_trainer.erpgs.config import ErpGSConfig
from ltx_trainer.erpgs.erp_projection import erp_lon_lat, erp_pixel_coords
from ltx_trainer.erpgs.erpgs_net import ErpGSStub
from ltx_trainer.erpgs.metrics import psnr, ssim_stub
from ltx_trainer.erpgs.synthetic import synthetic_batch


def evaluation_demo_run(cfg: ErpGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ErpGSConfig()
    rgb, mask, weight = synthetic_batch(cfg)
    model = ErpGSStub(cfg)
    with torch.no_grad():
        out = model(rgb, mask=mask, iteration=cfg.reg_start_iter + 1)
    mu = torch.tensor([[0.0, 0.0, 1.0]])
    lon, lat = erp_lon_lat(mu)
    uv = erp_pixel_coords(lon, lat, cfg.height, cfg.width)
    barber = TABLE1_NVS["OmniBlender"]["barbershop"]
    return {
        "erp_uv_center": [float(uv[0, 0]), float(uv[0, 1])],
        "weight_mean": float(weight.mean()),
        "mask_valid_frac": float(mask.mean()),
        "loss_L": float(out["L"].item()),
        "pred_shape": list(out["pred_rgb"].shape),
        "ours_psnr_barbershop": barber["Ours"]["PSNR"],
        "beats_omnigs_barbershop": ours_beats_baseline(barber, "OmniGS"),
    }


def train_step(cfg: ErpGSConfig | None = None, *, iteration: int = 12_000) -> dict[str, float]:
    cfg = cfg or ErpGSConfig()
    model = ErpGSStub(cfg)
    rgb, mask, _ = synthetic_batch(cfg)
    out = model(rgb, mask=mask, iteration=iteration)
    out["L"].backward()
    scalars: dict[str, float] = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            scalars[k] = float(v.detach())
    return scalars


def nvs_metrics_demo(cfg: ErpGSConfig | None = None) -> dict[str, float]:
    cfg = cfg or ErpGSConfig()
    gt, _, _ = synthetic_batch(cfg)
    pred = gt + 0.01 * torch.randn_like(gt)
    return {
        "psnr": float(psnr(pred, gt).item()),
        "ssim": float(ssim_stub(pred, gt).item()),
    }


def ablation_table_check() -> dict[str, bool]:
    all_row = TABLE2_ABLATION["All"]
    return {
        "all_best_psnr": all_row["PSNR"] > TABLE2_ABLATION["w/o W"]["PSNR"],
        "all_best_ssim": all_row["SSIM"] > TABLE2_ABLATION["w/o L_dn"]["SSIM"],
        "w_without_W_worst": TABLE2_ABLATION["w/o W"]["PSNR"] < TABLE2_ABLATION["w/o L_s"]["PSNR"],
    }
