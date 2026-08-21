"""X2HDR evaluation smoke — computed PU21 roundtrip + flow loss."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.x2hdr.benchmarks import TABLE2_ARRI, TABLE2_UPIQ
from ltx_trainer.x2hdr.model import X2Hdr, X2HdrConfig
from ltx_trainer.x2hdr.pipeline import evaluation_demo, pipeline_demo
from ltx_trainer.x2hdr.pu21_codec import scene_linear_to_vae_pixels
from ltx_trainer.x2hdr.synthetic import synthesize_hdr_scene, synthesize_sdr_from_hdr


def evaluation_smoke(*, device: str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    hdr = synthesize_hdr_scene(48, 48).to(dev)
    sdr = synthesize_sdr_from_hdr(hdr).to(dev)
    model = X2Hdr(X2HdrConfig()).to(dev)
    model.eval()
    with torch.no_grad():
        pred = model.infer(sdr, steps=5)
    pu21_gt = scene_linear_to_vae_pixels(hdr)
    pu21_pr = scene_linear_to_vae_pixels(pred.clamp(min=0.0))
    pu21_mae = float((pu21_gt - pu21_pr).abs().mean())
    model.train()
    loss, train_m = model.training_step(sdr, hdr)
    pipe = pipeline_demo(device=dev)
    eval_d = evaluation_demo(device=dev)
    return {
        "pu21_mae": pu21_mae,
        "loss_flow": train_m["loss_flow"],
        "loss_total": float(loss.detach()),
        "pipeline_pu21_l1": pipe["pu21_l1"],
        "eval_grad_norm": eval_d["grad_norm"],
        "table2_arri_anchor_pu21_psnr": TABLE2_ARRI["pu21_psnr"],
        "table2_upiq_anchor_pu21_psnr": TABLE2_UPIQ["pu21_psnr"],
        "computed_below_anchor": pu21_mae < 0.5,
    }
