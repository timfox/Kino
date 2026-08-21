"""Runnable evaluation smoke for World Tracing (arXiv:2606.13652)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.world_tracing.benchmarks import TABLE1_OBJECT_VISIBLE, benchmarks_bundle
from ltx_trainer.world_tracing.config import PAPER_ARXIV, WTConfig


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "world_tracing",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_wt_o_visible_mae": TABLE1_OBJECT_VISIBLE["WT-O"]["mae"],
        "ref_wt_o_full_l1": benchmarks_bundle()["table1_object_full"]["WT-O"]["l1"],
    }
    try:
        import torch

        from ltx_trainer.world_tracing.pipeline import evaluation_demo_run
        from ltx_trainer.world_tracing.representation import forward_fill_layers

        cfg = WTConfig(height=28, width=28, patch_size=14, width_dim=64, num_heads=2, num_blocks=1)
        demo = evaluation_demo_run(cfg, device="cpu")
        x = torch.randn(4, 8, 8, 3)
        valid = torch.zeros(4, 8, 8, dtype=torch.bool)
        valid[0] = True
        valid[2] = True
        filled = forward_fill_layers(x, valid)
        out.update(
            {
                "torch": True,
                "forward_fill_finite": bool(torch.isfinite(filled).all()),
                "demo_loss_fm": demo["train"]["loss_fm"],
                "demo_l0_mae": demo["train"]["l0_depth_mae"],
                "demo_voxels": demo["voxel_count"],
                "demo_fx": demo["intrinsics"]["fx"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
