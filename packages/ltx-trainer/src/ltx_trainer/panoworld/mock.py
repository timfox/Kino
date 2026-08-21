"""Runnable evaluation smoke for PanoWorld (arXiv:2605.13169)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld.benchmarks import TABLE2_PANOSPACE, benchmarks_bundle
from ltx_trainer.panoworld.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "panoworld",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_panospace_overall": TABLE2_PANOSPACE["overall"],
        "ref_bfov_miou": TABLE2_PANOSPACE["bfov_miou"],
    }
    try:
        import torch

        from ltx_trainer.panoworld.config import PanoWorldConfig
        from ltx_trainer.panoworld.erp_geometry import patch_spherical_directions, unit_ray
        from ltx_trainer.panoworld.panospace_metrics import bf_ov_iou
        from ltx_trainer.panoworld.pipeline import evaluation_demo_run
        from ltx_trainer.panoworld.ssca import SphericalSpatialCrossAttention

        cfg = PanoWorldConfig(height=48, width=96, hidden_dim=32, patch_size=16)
        demo = evaluation_demo_run(cfg, device="cpu")
        rays = patch_spherical_directions(48, 96, 16)
        assert torch.isfinite(rays).all()
        assert torch.isfinite(unit_ray(torch.tensor(0.0), torch.tensor(0.0))).all()
        ssca = SphericalSpatialCrossAttention(cfg)
        h0 = torch.randn(1, rays.shape[0], cfg.hidden_dim)
        enhanced = ssca(h0)
        iou = bf_ov_iou(
            torch.tensor([[0.0, 0.0, 40.0, 30.0]]),
            torch.tensor([[5.0, 2.0, 40.0, 30.0]]),
        )
        out.update(
            {
                "torch": True,
                "rays_finite": True,
                "ssca_shape_ok": enhanced.shape == h0.shape,
                "bf_ov_iou_sample": float(iou.item()),
                "demo_loss": demo["train"]["loss"],
                "demo_mc_acc": demo["eval"]["mc_acc"],
                "num_patches": demo["num_patches"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
