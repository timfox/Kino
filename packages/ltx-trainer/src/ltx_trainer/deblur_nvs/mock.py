"""Runnable evaluation smoke for DeblurNVS (arXiv:2606.01315)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deblur_nvs.benchmarks import TABLE2_DL3DV_BENCH, benchmarks_bundle, table2_ours
from ltx_trainer.deblur_nvs.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = table2_ours()
    out: dict[str, Any] = {
        "package": "deblur_nvs",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_lpips": ours["lpips"],
        "ref_fid": ours["fid"],
        "ref_time_s": ours["time"],
        "table2_methods": len(TABLE2_DL3DV_BENCH),
    }
    try:
        import torch

        from ltx_trainer.deblur_nvs.blur_synthesis import synthesize_blur_pair
        from ltx_trainer.deblur_nvs.config import DeblurNVSConfig
        from ltx_trainer.deblur_nvs.dataset import dataset_card
        from ltx_trainer.deblur_nvs.latent_models import DeblurNVSStub
        from ltx_trainer.deblur_nvs.pipeline import evaluation_demo_run, infer_novel_view
        from ltx_trainer.deblur_nvs.synthetic import synthetic_views

        cfg = DeblurNVSConfig(height=48, width=80, context_views=3)
        demo = evaluation_demo_run(cfg, device="cpu", seed=0)
        batch = synthetic_views(cfg, device="cpu", seed=1)
        model = DeblurNVSStub(cfg)
        sharp = torch.rand(3, 48, 80)
        _, blur, n = synthesize_blur_pair(sharp, seed=2)
        infer = infer_novel_view(model, batch["blur_context"], batch["camera"])
        out.update(
            {
                "torch": True,
                "blur_window": n,
                "dataset_pairs": dataset_card()["sharp_blur_pairs"],
                "demo_loss": demo["train"]["loss"],
                "novel_view_shape": list(infer["novel_view"].shape),
                "blur_sharp_delta": float((sharp - blur).abs().mean().item()),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
