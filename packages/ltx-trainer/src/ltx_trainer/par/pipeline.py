"""Training and evaluation demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.par.benchmarks import TABLE1_T2P, TABLE2_PO, TABLE3_CONSISTENCY, par_beats_ar_baseline
from ltx_trainer.par.circular_padding import circular_pad_width, circular_unpad_width
from ltx_trainer.par.config import PARConfig
from ltx_trainer.par.metrics import discontinuity_score, fid_stub
from ltx_trainer.par.par_net import PARStub
from ltx_trainer.par.synthetic import synthetic_erp, synthetic_text


def evaluation_demo_run(cfg: PARConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PARConfig()
    rgb = synthetic_erp(cfg)
    text = synthetic_text(cfg)
    model = PARStub(cfg)
    with torch.no_grad():
        out = model(rgb, text)
    padded = circular_pad_width(rgb, cfg.pad_ratio)
    unpadded = circular_unpad_width(padded, cfg.pad_ratio)
    return {
        "loss_L": float(out["L"].item()),
        "pred_shape": list(out["pred_rgb"].shape),
        "ds": float(discontinuity_score(rgb).item()),
        "pad_shape": list(padded.shape),
        "unpad_shape": list(unpadded.shape),
        "par_14b_fid": TABLE1_T2P["PAR-1.4B"]["FID"],
        "beats_panollama_fid": par_beats_ar_baseline("FID"),
        "po_with_prompt_fid": TABLE2_PO["PAR w/ prompt"]["FID"],
    }


def train_step(cfg: PARConfig | None = None) -> dict[str, float]:
    cfg = cfg or PARConfig()
    model = PARStub(cfg)
    out = model(synthetic_erp(cfg), synthetic_text(cfg))
    out["L"].backward()
    return {
        "L": float(out["L"].detach()),
        "L_va": float(out["L_va"].detach()),
        "L_consistency": float(out["L_consistency"].detach()),
    }


def ablation_checks() -> dict[str, bool]:
    return {
        "consistency_improves_fid": TABLE3_CONSISTENCY["w/ L_consistency"]["FID"]
        < TABLE3_CONSISTENCY["w/o L_consistency"]["FID"],
        "po_prompt_better": TABLE2_PO["PAR w/ prompt"]["FID"] < TABLE2_PO["PAR w/o prompt"]["FID"],
        "unified_tasks": TABLE1_T2P["PAR-0.3B"]["PO"] and TABLE1_T2P["PAR-0.3B"]["PE"],
    }


def metrics_demo(cfg: PARConfig | None = None) -> dict[str, float]:
    cfg = cfg or PARConfig()
    gt = synthetic_erp(cfg)
    pred = gt + 0.05 * torch.randn_like(gt)
    return {
        "fid_stub": float(fid_stub(gt, pred).item()),
        "ds_pred": float(discontinuity_score(pred).item()),
        "ds_gt": float(discontinuity_score(gt).item()),
    }
