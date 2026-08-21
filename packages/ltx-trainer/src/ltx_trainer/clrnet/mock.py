"""Runnable evaluation smoke for CLRNet (arXiv:2603.15767)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clrnet.benchmarks import TABLE2_ABLATION, benchmarks_bundle
from ltx_trainer.clrnet.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    full = next(r for r in TABLE2_ABLATION if "Proposed CLRNet" in r["config"])
    out: dict[str, Any] = {
        "package": "clrnet",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_cr_mae_cm": full["transl_cm"],
        "ref_cr_mae_rot": full["rot_deg"],
    }
    try:
        import torch

        from ltx_trainer.clrnet.clrnet_net import CLRNet, CRNet
        from ltx_trainer.clrnet.config import CLRNetConfig
        from ltx_trainer.clrnet.equirect import stack_lidar_depth, stack_radar_depth
        from ltx_trainer.clrnet.pipeline import evaluation_demo_run
        from ltx_trainer.clrnet.se3 import compose_transforms, transform_from_qt

        cfg = CLRNetConfig(height=48, width=96)
        demo = evaluation_demo_run(cfg, device="cpu")
        pts = torch.randn(1, 64, 3)
        pts[..., 2] = pts[..., 2].abs() + 1.0
        lidar_d = stack_lidar_depth(pts, torch.rand(1, 64), height=48, width=96)
        radar_d = stack_radar_depth(
            pts, torch.rand(1, 64), torch.randn(1, 64) * 0.1, torch.rand(1, 64), height=48, width=96
        )
        model = CLRNet(cfg)
        preds = model(torch.rand(1, 3, 48, 96), lidar_d, radar_d)
        q_cl, t_cl = preds["CL"]
        T_cl = transform_from_qt(q_cl, t_cl)
        T_loop = compose_transforms(T_cl, T_cl, T_cl)
        cr = CRNet(cfg)
        q_cr, t_cr = cr(torch.rand(1, 3, 48, 96), radar_d)
        out.update(
            {
                "torch": True,
                "demo_clr_loss": demo["clrnet"]["loss"],
                "loop_trace": float(T_loop[:, 0, 0].mean().item()),
                "cr_quat_norm": float(q_cr.norm(dim=-1).mean().item()),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
