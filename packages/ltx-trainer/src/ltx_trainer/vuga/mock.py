"""Runnable evaluation smoke for VUGA (arXiv:2604.23953)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuga.benchmarks import TABLE2_OIQA_VUGA, benchmarks_bundle
from ltx_trainer.vuga.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "vuga",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_JUFE_SRCC": TABLE2_OIQA_VUGA["JUFE-10K"]["SRCC"],
        "ref_OIQ_SRCC": TABLE2_OIQA_VUGA["OIQ-10K"]["SRCC"],
    }
    try:
        import torch

        from ltx_trainer.vuga.aff import SDAModule
        from ltx_trainer.vuga.cmp import CMPBlock
        from ltx_trainer.vuga.config import VUGAConfig
        from ltx_trainer.vuga.pipeline import evaluation_demo_run
        from ltx_trainer.vuga.vuga_net import VUGA

        cfg = VUGAConfig(input_size=64, stage_dims=(32, 64, 128, 256), cmp_dim=128)
        demo = evaluation_demo_run(cfg, device="cpu")
        x = torch.rand(2, 3, 64, 64)
        model = VUGA(cfg)
        scores = model(x)["score"]
        cmp_out = CMPBlock(32, 128)(torch.rand(1, 32, 16, 16))
        sda_out = SDAModule(128)(torch.rand(1, 128, 8, 8))
        out.update(
            {
                "torch": True,
                "scores_finite": bool(torch.isfinite(scores).all()),
                "cmp_shape": list(cmp_out.shape),
                "sda_shape": list(sda_out.shape),
                "demo_loss": demo["train"]["loss"],
                "demo_srcc": demo["train"].get("srcc", 0.0),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
