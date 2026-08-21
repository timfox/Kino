"""Training / eval pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.cross360.benchmarks import table1_ours_m3d, table2_ours_struct3d
from ltx_trainer.cross360.config import Cross360Config
from ltx_trainer.cross360.cross360_net import Cross360NetStub
from ltx_trainer.cross360.losses import total_depth_loss
from ltx_trainer.cross360.synthetic import synthetic_batch
from ltx_trainer.cross360.tangent import tp_sampling_layout


def train_step(cfg: Cross360Config | None = None) -> dict[str, float]:
    cfg = cfg or Cross360Config(height=64, width=128)
    model = Cross360NetStub(cfg)
    model.train()
    erp, depth_gt = synthetic_batch(cfg)
    out = model(erp)
    preds = [out["depth"], *out["depth_scales"]]
    targets = [
        F.interpolate(depth_gt, size=p.shape[-2:], mode="bilinear", align_corners=False) for p in preds
    ]
    losses = total_depth_loss(preds, targets)
    losses["L_total"].backward()
    return {k: float(v.detach()) for k, v in losses.items()}


def evaluation_demo_run(cfg: Cross360Config | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or Cross360Config(height=64, width=128)
    model = Cross360NetStub(cfg).to(device)
    model.eval()
    erp, _ = synthetic_batch(cfg, batch_size=1)
    erp = erp.to(device)
    with torch.no_grad():
        out = model(erp)
    return {
        "depth_shape": list(out["depth"].shape),
        "num_scales": len(out["depth_scales"]),
        "tp_layout": tp_sampling_layout(incomplete_fov=cfg.incomplete_fov),
        "reference_m3d_AbsRel": table1_ours_m3d()["AbsRel"],
        "reference_struct3d_AbsRel": table2_ours_struct3d()["AbsRel"],
    }
