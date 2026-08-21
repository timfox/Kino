"""Training / ablation demos."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.prior_flow.benchmarks import TABLE1_ABLATION, TABLE7_REGIONS
from ltx_trainer.prior_flow.config import PriorFlowConfig
from ltx_trainer.prior_flow.distortion import orthogonal_distortion_map, primitive_distortion_map
from ltx_trainer.prior_flow.losses import iterative_flow_loss
from ltx_trainer.prior_flow.orthogonal_view import primitive_to_orthogonal
from ltx_trainer.prior_flow.prior_flow_net import PriorFlowStub
from ltx_trainer.prior_flow.synthetic import synthetic_erp_pair, synthetic_flow_gt


def evaluation_demo_run(cfg: PriorFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PriorFlowConfig(height=64, width=128, num_iterations=3)
    model = PriorFlowStub(cfg)
    f1, f2 = synthetic_erp_pair(cfg)
    with torch.no_grad():
        out = model(f1, f2)
    return {
        "flow_p_shape": list(out["flow_primitive"].shape),
        "flow_o_shape": list(out["flow_orthogonal"].shape),
        "num_iters": len(out["predictions_primitive"]),
    }


def ablation_modules(cfg: PriorFlowConfig | None = None) -> dict[str, float]:
    cfg = cfg or PriorFlowConfig(height=48, width=96, num_iterations=2)
    f1, f2 = synthetic_erp_pair(cfg)
    full = PriorFlowStub(cfg)
    no_dccl = PriorFlowStub(replace(cfg, use_dccl=False))
    no_oddc = PriorFlowStub(replace(cfg, use_oddc=False))
    with torch.no_grad():
        a = full(f1, f2)["flow_primitive"].mean()
        b = no_dccl(f1, f2)["flow_primitive"].mean()
        c = no_oddc(f1, f2)["flow_primitive"].mean()
    return {
        "full_mean": float(a),
        "no_dccl_mean": float(b),
        "no_oddc_mean": float(c),
        "table1_full_epe": TABLE1_ABLATION["PriOr_RAFT_full"]["all_epe"],
        "table1_poles_epe": TABLE1_ABLATION["PriOr_RAFT_full"]["poles_epe"],
    }


def distortion_demo(height: int = 64, width: int = 128) -> dict[str, Any]:
    prim = primitive_distortion_map(height, width)
    orth = orthogonal_distortion_map(height, width)
    pole = prim[:, :, 0, :].mean()
    equator = prim[:, :, height // 2, :].mean()
    return {
        "equator_higher_than_pole": float(equator > pole),
        "orthogonal_pole_higher": float(orth[:, :, 0, :].mean() > orth[:, :, height // 2, :].mean()),
    }


def orthogonal_demo(cfg: PriorFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PriorFlowConfig(height=32, width=64)
    f1, _ = synthetic_erp_pair(cfg)
    orth = primitive_to_orthogonal(f1)
    return {"primitive_shape": list(f1.shape), "orthogonal_shape": list(orth.shape)}


def train_step(cfg: PriorFlowConfig | None = None) -> dict[str, float]:
    cfg = cfg or PriorFlowConfig(height=48, width=96, num_iterations=3)
    model = PriorFlowStub(cfg)
    f1, f2 = synthetic_erp_pair(cfg)
    gt = synthetic_flow_gt(cfg)
    out = model(f1, f2)
    loss_p = iterative_flow_loss(out["predictions_primitive"], gt)
    loss_o = iterative_flow_loss(out["predictions_orthogonal"], gt)
    return {
        "loss_primitive": float(loss_p.detach()),
        "loss_orthogonal": float(loss_o.detach()),
        "loss_total": float((loss_p + loss_o).detach()),
        "table7_poles_gain": TABLE7_REGIONS["PanoFlow"]["poles_epe"] / TABLE7_REGIONS["PriOr_RAFT"]["poles_epe"],
    }
