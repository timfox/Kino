"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.ulayout.benchmarks import (
    TABLE3_MATTERPORT_LSUN,
    TABLE4_ABLATION,
    efficient_reduces_flops,
    joint_training_wins_ablation,
    ours_beats_lsun_room,
    ours_beats_pano_baseline,
)
from ltx_trainer.ulayout.config import ULayoutConfig
from ltx_trainer.ulayout.metrics import iou_2d_floor_stub, iou_3d_room_stub
from ltx_trainer.ulayout.ulayout_net import ULayoutStub


def _synthetic_batch(h: int = 64, pano_w: int = 128, pp_w: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pano = torch.rand(1, 3, h, pano_w)
    pp = torch.rand(1, 3, h, pp_w)
    tgt_pano = torch.stack([torch.rand(1, h, pano_w), torch.rand(1, h, pano_w)], dim=1)
    tgt_pp = torch.rand(1, h, pp_w)
    return pano, pp, tgt_pano, tgt_pp


def evaluation_demo_run(cfg: ULayoutConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ULayoutConfig()
    model = ULayoutStub(cfg)
    pano, pp, tgt_p, tgt_pp = _synthetic_batch()
    with torch.no_grad():
        out = model(pano, pp, pitch_deg=5.0, tgt_pano_b=tgt_p, tgt_pp_b=tgt_pp)
    pred_c = out["ceiling"]
    tgt_c = tgt_p[:, 0]
    if pred_c.shape != tgt_c.shape:
        pred_c = F.interpolate(
            pred_c.unsqueeze(1), size=tgt_c.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    iou2 = float(iou_2d_floor_stub(pred_c, tgt_c).mean())
    iou3 = float(iou_3d_room_stub(out["ceiling"]).mean())
    return {
        "ceiling_shape": list(out["ceiling"].shape),
        "loss": float(out["loss"].item()),
        "iou_2d_proxy": iou2,
        "iou_3d_proxy": iou3,
        "table3_ours_pano_2d": TABLE3_MATTERPORT_LSUN["Ours"]["pano_2d_iou"],
        "beats_lgt_pano": ours_beats_pano_baseline(TABLE3_MATTERPORT_LSUN, "LGT-Net"),
    }


def train_step(cfg: ULayoutConfig | None = None) -> dict[str, float]:
    cfg = cfg or ULayoutConfig()
    model = ULayoutStub(cfg)
    pano, pp, tgt_p, tgt_pp = _synthetic_batch()
    out = model(pano, pp, pitch_deg=0.0, tgt_pano_b=tgt_p, tgt_pp_b=tgt_pp)
    out["loss"].backward()
    return {"loss": float(out["loss"].detach())}


def ablation_table_check() -> dict[str, bool]:
    return {
        "joint_beats_single_domain": joint_training_wins_ablation(),
        "ours_beats_lsun_room_t3": ours_beats_lsun_room(TABLE3_MATTERPORT_LSUN),
        "efficient_flops_reduction": efficient_reduces_flops(),
        "vertical_shift_matters": TABLE4_ABLATION["Ours"]["lsun_ceiling"]
        > TABLE4_ABLATION["w/o_vertical_shift"]["lsun_ceiling"],
    }
