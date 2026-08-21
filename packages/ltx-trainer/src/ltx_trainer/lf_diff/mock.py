"""LF-Diff evaluation smoke — computed tonemap + LPR losses."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lf_diff.benchmarks import TABLE2_REAL
from ltx_trainer.lf_diff.model import LfDiff, LfDiffConfig
from ltx_trainer.lf_diff.pipeline import evaluation_demo, pipeline_demo
from ltx_trainer.lf_diff.synthetic import synthesize_exposure_bracket, synthesize_hdr_scene
from ltx_trainer.lf_diff.tonemap import tonemap_l1


def evaluation_smoke(*, device: str = "cpu") -> dict[str, Any]:
    dev = torch.device(device)
    hdr = synthesize_hdr_scene(48, 48).to(dev)
    brackets = [b.to(dev) for b in synthesize_exposure_bracket(hdr)]
    model = LfDiff(LfDiffConfig()).to(dev)
    model.eval()
    with torch.no_grad():
        pred = model.infer(brackets, steps=5)
    tm_mae = float(tonemap_l1(pred, hdr))
    model.train()
    loss, train_m = model.training_step(brackets, hdr)
    pipe = pipeline_demo(device=dev)
    eval_d = evaluation_demo(device=dev)
    anchor = TABLE2_REAL["lf_diff"]["pu21_psnr"]
    return {
        "tonemap_mae": tm_mae,
        "loss_tonemap": train_m["loss_tonemap"],
        "loss_lpr": train_m["loss_lpr"],
        "loss_total": float(loss.detach()),
        "pipeline_tonemap_l1": pipe["tonemap_l1"],
        "eval_grad_norm": eval_d["grad_norm"],
        "table2_anchor_pu21_psnr": anchor,
        "computed_finite": bool(torch.isfinite(loss)),
    }
