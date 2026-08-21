"""Runnable evaluation smoke for PanoAffordanceNet (arXiv:2603.09760)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_affordance.benchmarks import TABLE1_360_AGD, benchmarks_bundle
from ltx_trainer.pano_affordance.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = next(r for r in TABLE1_360_AGD if r["method"] == "Ours")
    out: dict[str, Any] = {
        "package": "pano_affordance",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_easy_kld": ours["easy"]["KLD"],
        "ref_hard_nss": ours["hard"]["NSS"],
    }
    try:
        import torch

        from ltx_trainer.pano_affordance.config import PanoAffordanceConfig
        from ltx_trainer.pano_affordance.dasm import DASM
        from ltx_trainer.pano_affordance.osdh import OSDH
        from ltx_trainer.pano_affordance.pano_affordance_net import PanoAffordanceNet
        from ltx_trainer.pano_affordance.pipeline import evaluation_demo_run

        cfg = PanoAffordanceConfig(height=56, width=112, num_classes=8)
        demo = evaluation_demo_run(cfg, device="cpu")
        model = PanoAffordanceNet(cfg)
        img = torch.rand(1, 3, 56, 112)
        o = model(img)
        out.update(
            {
                "torch": True,
                "demo_loss": demo["train"]["loss"],
                "map_shape": list(o["a_refined"].shape),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
