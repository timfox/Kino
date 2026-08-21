"""Protocol demos for OB3D benchmarks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ob3d.benchmarks import TABLE3_CPE, TABLE4_NVS, TABLE5_RECON
from ltx_trainer.ob3d.cameras import (
    camera_forward_from_positions,
    rotation_world_to_camera,
    synthetic_egocentric_positions,
    synthetic_non_egocentric_positions,
)
from ltx_trainer.ob3d.config import OB3DConfig, eval_indices, train_indices
from ltx_trainer.ob3d.metrics import (
    ate_rmse,
    auc_at_threshold,
    psnr,
    relative_rotation_error,
    relative_translation_error,
    ssim_stub,
)
from ltx_trainer.ob3d.recon_stub import evaluate_depth_reconstruction
from ltx_trainer.ob3d.synthetic import synthetic_depth, synthetic_normal, synthetic_rgb


def evaluation_demo_run(cfg: OB3DConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OB3DConfig()
    return {
        "rgb_shape": list(synthetic_rgb(cfg).shape),
        "depth_shape": list(synthetic_depth(cfg).shape),
        "normal_shape": list(synthetic_normal(cfg).shape),
        "train_indices_head": train_indices()[:5],
        "eval_indices_head": eval_indices()[:5],
    }


def cpe_demo() -> dict[str, float]:
    pos = synthetic_egocentric_positions(20)
    fwd = camera_forward_from_positions(pos)
    r_gt = rotation_world_to_camera(fwd[0])
    r_est = rotation_world_to_camera(fwd[1])
    rra = relative_rotation_error(r_est, r_gt)
    rta = relative_translation_error(pos[1] - pos[0], pos[2] - pos[1])
    ate = ate_rmse(pos[1:], pos[1:] + 0.01 * torch.randn_like(pos[1:]))
    return {
        "RRA_deg": rra,
        "RTA_deg": rta,
        "AUC5_stub": auc_at_threshold([rra], [rta]),
        "ATE": ate,
        "table3_outdoor_non_ego_RRA": TABLE3_CPE["outdoor_OpenMVG_non_ego"]["RRA"],
    }


def nvs_demo(cfg: OB3DConfig | None = None) -> dict[str, float]:
    cfg = cfg or OB3DConfig()
    gt = synthetic_rgb(cfg)
    pred = gt + 0.02 * torch.randn_like(gt)
    return {
        "PSNR": float(psnr(pred, gt).item()),
        "SSIM": float(ssim_stub(pred, gt).item()),
        "table4_OmniGS_indoor_PSNR": TABLE4_NVS["OmniGS_indoor_ego"]["PSNR"],
    }


def recon_demo(cfg: OB3DConfig | None = None) -> dict[str, float]:
    metrics = evaluate_depth_reconstruction(cfg)
    metrics["table5_NeuS_all_non_ego_RMSE"] = TABLE5_RECON["NeuS_all_non_ego"]["RMSE"]
    return metrics


def trajectory_demo() -> dict[str, Any]:
    ego = synthetic_egocentric_positions(10)
    non_ego = synthetic_non_egocentric_positions(10)
    return {
        "egocentric_positions": ego.shape,
        "non_egocentric_positions": non_ego.shape,
    }


def train_step(cfg: OB3DConfig | None = None) -> dict[str, float]:
    from ltx_trainer.ob3d.recon_stub import DepthReconStub

    cfg = cfg or OB3DConfig()
    model = DepthReconStub(cfg)
    rgb = synthetic_rgb(cfg)
    gt = synthetic_depth(cfg)
    pred = model(rgb)
    loss = torch.nn.functional.l1_loss(pred, gt)
    return {"depth_l1": float(loss.detach())}
